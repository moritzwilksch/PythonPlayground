#%%
from code import compile_command
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

#%%
# import dask.dataframe as dd
# from dask.distributed import Client
# client = Client()


#%%
df = pd.read_parquet("data/yellow_tripdata_2021-01.parquet")


#%%
def fix_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    CATEGORICALS = [
        "store_and_fwd_flag",
        "VendorID",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "payment_type",
    ]
    data[CATEGORICALS] = data[CATEGORICALS].astype("category")
    return data


def drop_outliers(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(
        trip_duration=lambda d: d["tpep_dropoff_datetime"] - d["tpep_pickup_datetime"]
    )

    data = data.loc[data["trip_duration"] >= pd.Timedelta(1, "min")]
    data = data.query("trip_distance > 0")
    data = data.query("total_amount > 0")
    return data


def fix_negative_dollar_amounts(data: pd.DataFrame) -> pd.DataFrame:
    COLUMNS = ["total_amount"]
    for col in COLUMNS:
        data[col] = np.where(data[col] < 0, data[col] * -1, data[col])
    return data


#%%
clean = df.copy().pipe(fix_dtypes).pipe(drop_outliers).pipe(fix_negative_dollar_amounts)
print(f"Dropped {len(df) - len(clean)} rows.")

#%%
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def split_data(clean: pd.DataFrame) -> tuple:
    X = clean.drop(
        [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "tip_amount",
            "total_amount",
            "trip_duration",
            "fare_amount",
            "extra",
            "mta_tax",
            "tolls_amount",
            "improvement_surcharge",
            "congestion_surcharge",
        ],
        axis=1,
    )
    y = clean["total_amount"]

    xtrainval, xtest, ytrainval, ytest = train_test_split(X, y, random_state=42)
    xtrain, xval, ytrain, yval = train_test_split(xtrainval, ytrainval, random_state=42)

    return X, y, xtrain, ytrain, xval, yval, xtest, ytest


X, y, xtrain, ytrain, xval, yval, xtest, ytest = split_data(clean)
#%%
from lightgbm.callback import early_stopping

model = LGBMRegressor(n_estimators=1_000, num_leaves=32, objective="mae")
model.fit(
    xtrain,
    ytrain,
    eval_set=(xval, yval),
    eval_metric=["mape", "mae"],
    callbacks=[early_stopping(500)],
    verbose=50,
)

#%%
val_preds = model.predict(xval)
plt.scatter(yval, val_preds)

#%%
from lightgbm.plotting import plot_importance

plot_importance(model)

#%%
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

test_preds = model.predict(xtest)
print(mean_absolute_error(ytest, test_preds))
print(mean_absolute_percentage_error(ytest, test_preds))

#%%
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model,
    X=xval.sample(10000),
    features=["trip_distance"],
    grid_resolution=20,
    kind="individual",
)

#%%
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer

# Linear Model
correct_congestion_surcharge = clean.eval(
    "total_amount - improvement_surcharge - tolls_amount - tip_amount - mta_tax - extra - fare_amount"
).round(1)

imputation_condition = clean["congestion_surcharge"].isna() & (
    clean.eval(
        "improvement_surcharge + tolls_amount + tip_amount + mta_tax + extra + fare_amount"
    ).round(2)
    + correct_congestion_surcharge
    == clean["total_amount"]
)
clean_nona = clean.assign(
    airport_fee=lambda d: d["airport_fee"].fillna(0),
    congestion_surcharge=np.where(
        imputation_condition,
        correct_congestion_surcharge,
        clean["congestion_surcharge"],
    ),
)
clean_nona = clean_nona.assign(
    passenger_count=lambda d: d["passenger_count"].ffill(),
    store_and_fwd_flag=lambda d: d["store_and_fwd_flag"].ffill(),
)

# X, y, xtrain, ytrain, xval, yval, xtest, ytest = split_data(clean_nona)

# impute_transforms = ColumnTransformer(
#     [
#         ("ss", StandardScaler(), X.select_dtypes(np.number).columns),
#         ("ohe", OneHotEncoder(sparse=False), X.select_dtypes("category").columns),
#     ]
# )

# impute_pipeline = Pipeline([("transforms", impute_transforms), ("impute", KNNImputer())])


# xtrain_nona = impute_pipeline.fit_transform(xtrain, ytrain)
# xval_nona = impute_pipeline.transform(xval)


#%%
from scipy import stats

stats.chi2_contingency(
    clean["RatecodeID"].value_counts().values,
    clean["RatecodeID"].ffill().value_counts().values,
)

#%%


#%%


# TODO
# - correlation of explicit missingness


#%%
# df2 = dd.from_pandas(df, npartitions=6)

# #%%
# df2.corr().compute()

# #%%``
# client.shutdown()

#%%
@np.vectorize
def poisson_deviance(y, yhat):
    if y == 0 or yhat == 0:
        log_term = 0
    else:
        log_term = np.log(y / yhat)

    return y * log_term - (y - yhat)


a = np.array([1])
b = np.array([4])

print(poisson_deviance(a, b))

#%%
x, y = np.meshgrid(np.arange(0, 10, 0.1), np.arange(0, 10, 0.1))
z = poisson_deviance(x, y)

#%%
fig, ax = plt.subplots(figsize=(6, 5))
cf = ax.contourf(x, y, z, levels=50, cmap="inferno")
fig.colorbar(cf)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)


#%%
x = np.arange(0, 5, 0.1)
plt.plot(x, poisson_deviance(np.arange([1]), x))


#%%
df2 = pl.from_pandas(df)

compcols = df2.select(pl.col(pl.Float64).exclude("passenger_count")).columns

df2.select(
    [pl.spearman_rank_corr(pl.col("passenger_count"), compcol).alias(f"_{compcol}") for compcol in compcols]
).transpose(include_header=True).sort(by="column_0")

#%%