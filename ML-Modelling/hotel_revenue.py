#%%
from ast import Or
from multiprocessing.connection import Pipe
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_excel("data/COVID19_time_series.xlsx")

#%%
def fix_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    data.columns = [c.lower() for c in data.columns]
    data = data.assign(location=lambda d: d["location"].astype("category"))
    return data


def equalize_start_date(data: pd.DataFrame) -> pd.DataFrame:
    return data.query("date >= '2013-01-01'")


def engineer_lagged_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.assign(day_of_week=data["date"].dt.day_of_week)

    LAG_CONFIG = {
        "revenue": 7,
        "demand": 7,
        "occupancy": 7,
    }

    for lagged_ft, max_lag in LAG_CONFIG.items():
        for lag in range(1, max_lag + 1):
            data = data.assign(
                **{
                    f"{lagged_ft}_lag_{lag}": data.groupby("location")[lagged_ft].shift(
                        lag
                    )
                }
            )

    return data


clean = df.pipe(fix_dtypes).pipe(equalize_start_date).pipe(engineer_lagged_features)

#%%
clean["location"].value_counts()

#%%
clean.groupby("location")["date"].agg(["min", "max"])

#%%
fig, ax = plt.subplots(figsize=(20, 8))
sns.lineplot(
    data=clean,  # .query("'2018-01-01' < date < '2018-02-01'"),
    x="date",
    y="revenue",
    hue="location",
    ax=ax,
)

#%%
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

transformer = ColumnTransformer(
    [("oe", OrdinalEncoder(), ["location"]), ("drop_date", "drop", ["date", "day"])],
    remainder="passthrough",
    verbose_feature_names_out=False,
)

pipeline = Pipeline(
    [
        ("transformer", transformer),
        (
            "make_df",
            FunctionTransformer(
                lambda d: pd.DataFrame(d, columns=transformer.get_feature_names_out())
            ),
        ),
        ("model", LGBMRegressor(categorical_features=[0])),
    ]
)

# tss = TimeSeriesSplit()

nocovid_subset = clean.query("'2013-01-01' <= date <= '2020-01-01'")
X = nocovid_subset.drop(["revenue", "occupancy", "demand"], axis=1)
y = nocovid_subset["revenue"]

DATE_THRESH = "2018-01-01"

xtrain, xval = X.query("date < @DATE_THRESH"), X.query("date >= @DATE_THRESH")
ytrain, yval = y[xtrain.index], y[xval.index]

#%%
pipeline.fit(xtrain, ytrain)

#%%
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

preds = pipeline.predict(xval)
print(
    f"MAE = {mean_absolute_error(yval, preds):,.1f}, MSE = {mean_squared_error(yval, preds):,.1f}, MAPE = {mean_absolute_percentage_error(yval, preds):.4f}"
)

#%%

xval_subset = clean.query("location == 'NewYork'").query("date > '2020-01-01'")
yval = clean.iloc[xval_subset.index]["revenue"]
yval_subset = yval[xval_subset.index]
preds = pipeline.predict(xval_subset)

fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(xval_subset.date, yval_subset, color="k")
ax.plot(xval_subset.date, preds, color="blue")

#%%
from lightgbm import plot_importance

plot_importance(pipeline["model"])
