#%%
import imp
from sklearn.metrics import mean_squared_error, mean_absolute_error
from random import random
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv("data/AB_NYC_2019.csv")
df.head()

#%%
# data cleaning pipeline steps


def drop_unused_cols(data: pd.DataFrame) -> pd.DataFrame:
    # TODO: use textual features like "name": positive words might be predictive of price
    UNUSED = ["id", "name", "host_id", "host_name"]
    return data.drop(UNUSED, axis=1)


def drop_zero_price_rows(data: pd.DataFrame) -> pd.DataFrame:
    n_before = len(data)
    data = data.loc[data["price"] > 0]
    n_after = len(data)
    print(f"INFO: dropped {n_before - n_after} rows with zero price")
    return data


def drop_outliers(data: pd.DataFrame) -> pd.DataFrame:
    n_before = len(data)
    data = data.loc[data["price"] <= np.quantile(data["price"], 0.99)]
    data = data.loc[data["minimum_nights"] <= np.quantile(data["minimum_nights"], 0.99)]
    n_after = len(data)
    print(f"INFO: dropped {n_before - n_after} outliers")
    return data


def merge_rare_neighborhoods(data: pd.DataFrame) -> pd.DataFrame:
    n_occurences = df["neighbourhood"].value_counts()
    rare_neigborhoods = n_occurences[n_occurences < 10].index.to_list()
    return data.assign(
        neighbourhood=data["neighbourhood"].replace(rare_neigborhoods, "RareNeighborhood")
    )


def transform_response(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(log_price=np.log(data["price"])).drop("price", axis=1)
    # return data.assign(log_price=data["price"]).drop("price", axis=1)


def fix_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    CATCOLS = ["neighbourhood_group", "neighbourhood", "room_type"]
    DATECOL = "last_review"

    data[CATCOLS] = data[CATCOLS].astype("category")
    data[DATECOL] = pd.to_datetime(data[DATECOL])
    return data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(
        days_since_last_review=(data["last_review"].max() - data["last_review"]).dt.days
    ).drop("last_review", axis=1)
    return data


clean = (
    df.copy()
    .pipe(drop_unused_cols)
    .pipe(drop_zero_price_rows)
    .pipe(drop_outliers)
    .pipe(merge_rare_neighborhoods)
    .pipe(transform_response)
    .pipe(fix_dtypes)
    .pipe(feature_engineering)
)

#%%
def get_x_y(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = data.drop(["log_price"], axis=1)
    y = data["log_price"]
    return X, y


X, y = get_x_y(clean)


def get_train_val_test_data(X, y):
    x_train, x_valtest, y_train, y_valtest = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_valtest, y_valtest, test_size=0.5, random_state=42
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


x_train, x_val, x_test, y_train, y_val, y_test = get_train_val_test_data(X, y)
print(f"train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}")


#%%


def eval_preds(heading, y_true, y_pred):
    print(f"{heading:*^80}")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print("log-scale:")
    print(f"\tRMSE: {rmse:.4f}, MAE: {mae:.4f}")

    y_true, y_pred = np.exp(y_true), np.exp(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print("original scale:")
    print(f"\tRMSE: {rmse:.4f}, MAE: {mae:.4f}")


#%%
# Set up a baseline model
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
dummy.fit(x_train, y_train)
eval_preds("Dummy Model", y_val, dummy.predict(x_val))

#%%
from lightgbm import LGBMRegressor

model = LGBMRegressor(n_estimators=1000)
model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=50)

#%%
eval_preds("LGBM Vanilla", y_val, model.predict(x_val))

#%%
import optuna


def fit_model(params, x_train, y_train, x_val, y_val):
    model = LGBMRegressor(n_estimators=1000, **params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=50, verbose=0)
    return model


def objective(trial: optuna.Trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 4, 128, 4),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-10, 100),
    }
    model = fit_model(params, x_train, y_train, x_val, y_val)
    val_preds = model.predict(x_val)
    return mean_squared_error(y_val, val_preds)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

#%%
model = fit_model(study.best_params, x_train, y_train, x_val, y_val)
eval_preds("LGBM Optuna", y_val, model.predict(x_val))
#%%
preds = model.predict(x_val)
plt.scatter(y_val, preds, color="k", alpha=0.2)
plt.plot([3, 7], [3, 7], color="r")
#%%
plt.scatter(y_val, preds - y_val, color="k", alpha=0.2)


#%%
from lightgbm.plotting import plot_importance
plot_importance(model, max_num_features=10, importance_type="gain")

#%%
import duckdb

duckdb.query(
    """ 
    WITH words as (
        SELECT unnest(split(lower(name), ' ')) as word
        FROM df
        )
    SELECT word, count(*) as n FROM words GROUP BY word ORDER BY n DESC

 """
).df().head(25)
