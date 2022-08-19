#%%
from ast import Or
from timeit import timeit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

#%%
store = pd.read_csv(
    "https://github.com/moritzwilksch/RossmannSalesPrediction/raw/main/data/store.csv"
)
train = pd.read_csv(
    "https://github.com/moritzwilksch/RossmannSalesPrediction/raw/main/data/train.csv"
)

#%%
print(store["Assortment"].value_counts().agg(["count", "min", "max"]))
print(store["Assortment"].nunique())
print(store["Assortment"].isna().sum())

#%%
sns.histplot(store["CompetitionDistance"])

#%%
def cast_categoricals(data: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    if cols is None:
        cols = ["Assortment", "StoreType", "PromoInterval"]

    data[cols] = data[cols].astype("category")
    return data


def create_competition_open_since(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(
        CompOpenSince=lambda d: pd.to_datetime(
            {
                "year": data["CompetitionOpenSinceYear"],
                "month": data["CompetitionOpenSinceMonth"],
                "day": 1,
            }
        )
    ).drop(["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"], axis=1)


def create_promo2_since(data: pd.DataFrame) -> pd.DataFrame:

    return data.assign(
        Promo2Since=lambda d: pd.to_datetime(
            data["Promo2SinceWeek"].astype("Int64").astype(str)
            + data["Promo2SinceYear"].astype("Int64").astype(str).add("-1"),
            format="%V%G-%u",
            errors="coerce",
        ),
    ).drop(["Promo2SinceWeek", "Promo2SinceYear"], axis=1)


clean = (
    store.copy()
    .pipe(cast_categoricals)
    .pipe(create_competition_open_since)
    .pipe(create_promo2_since)
)

#%%
sns.barplot(data=train, x="DayOfWeek", y="Sales", ci=None)

#%%
sns.scatterplot(data=train, x="Customers", y="Sales")

#%%


def fix_train_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    data["Date"] = data["Date"].astype("datetime64")
    data["StateHoliday"] = data["StateHoliday"].astype("bool")
    data["SchoolHoliday"] = data["SchoolHoliday"].astype("bool")
    return data


train = train.pipe(fix_train_dtypes)

X = pd.merge(train, clean, how="left", on="Store")
X

#%%
def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = pl.from_pandas(data)
    # on promo indicator
    data = data.with_column(
        pl.col("PromoInterval")
        .cast(pl.Utf8)
        .str.split(",")
        .arr.contains(pl.col("Date").dt.strftime("%b"))
        .alias("on_promo"),
    )

    data = data.with_column(
        (pl.col("Date") - pl.col("CompOpenSince")).alias("CompOpenSinceDays"),
    )

    return data.to_pandas()


X = X.pipe(add_features).query("Sales > 0")

#%%
x = X[
    [
        "DayOfWeek",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "StoreType",
        "Assortment",
        # "CompetitionDistance",
        "Promo2",
        # "PromoInterval",
        # "on_promo",
    ]
]
y = X["Sales"]

#%%
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, GammaRegressor, TweedieRegressor
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()
x.loc[:, ["StoreType", "Assortment"]] = oe.fit_transform(x[["StoreType", "Assortment"]])

# model = GammaRegressor()
# model = LinearRegression()
model = TweedieRegressor(power=3)

model.fit(x, y)
pred = model.predict(x)
print(mean_absolute_error(y, pred))

#%%
store = pl.from_pandas(store)
#%%
store.select(
    pl.concat_str(
        [
            pl.col("CompetitionOpenSinceYear").cast(pl.Int16).cast(pl.Utf8),
            pl.lit("-"),
            pl.col("CompetitionOpenSinceMonth").cast(pl.Int16).cast(pl.Utf8),
            pl.lit("-01"),
        ]
    ).alias("date")
).select(pl.col("date").str.strptime(pl.Date, "%Y-%-m-%d"))
