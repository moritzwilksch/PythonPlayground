#%%
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#%%
import dask.dataframe as dd
from dask.distributed import Client
client = Client()


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
    ],
    axis=1,
)
y = clean["total_amount"]

xtrainval, xtest, ytrainval, ytest = train_test_split(X, y, random_state=42)
xtrain, xval, ytrain, yval = train_test_split(xtrainval, ytrainval, random_state=42)


#%%
from lightgbm.callback import early_stopping

model = LGBMRegressor(n_estimators=1_000, num_leaves=32, objective="mae")
# model.fit(
#     xtrain,
#     ytrain,
#     eval_set=(xval, yval),
#     eval_metric=["mape", "mae"],
#     callbacks=[early_stopping(500)],
#     verbose=50,
# )

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

# TODO
# - correlation of explicit missingness




#%%
df2 = dd.from_pandas(df, npartitions=6)

#%%
df2.corr().compute()

#%%``
client.shutdown()