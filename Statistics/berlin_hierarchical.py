#%%
from logging.config import valid_ident
from multiprocessing.sharedctypes import Value
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_parquet("data/berlin.parquet")

#%%
df.isna().sum()

#%%
df["object_type"].value_counts()

#%%


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    object_type_to_remove = ["RETIREMENT_HOME", "NURSING_HOME"]
    data = data.loc[~data["object_type"].isin(object_type_to_remove)].copy()
    data["object_type"] = data["object_type"].cat.remove_unused_categories()
    return data


def fix_factor_100(data: pd.DataFrame) -> pd.DataFrame:
    data[["price", "square_meters"]] = data[["price", "square_meters"]] / 100.0
    return data


@np.vectorize
def _fix_rooms_helper(s: str) -> float:
    valid_mapping = {
        "Privatzimmer": 1.0,
        "Gemeinsames Zimmer": 0.5,
    }

    x = np.nan
    try:
        x = float(s)
        if x > 10 or x == 0:
            x = np.nan
    except:
        if s in valid_mapping:
            x = valid_mapping[s]

    return x


def fix_rooms(data: pd.DataFrame) -> pd.DataFrame:
    data["rooms"] = data["rooms"].apply(_fix_rooms_helper)
    return data


def fix_sqm(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[data["square_meters"] == 0, "square_meters"] = np.nan
    return data


clean = df.copy().pipe(filter_data).pipe(fix_factor_100).pipe(fix_rooms).pipe(fix_sqm)

#%%
plt.hist(
    data=clean.query("(to_rent == False) & (price < 10_000_000)"), x="price", bins=100
)
#%%
plt.hist(data=clean.query("(to_rent == True) & (price < 10_000)"), x="price", bins=100)

#%%
clean["private_offer"].value_counts()


#%%
plt.hist(clean.query("square_meters < 2_000")["square_meters"], bins=100)


#%%
with pd.option_context("display.max_rows", 100):
    print(clean.query("square_meters > 1000").to_markdown())
#%%

clean["square_meters"].fillna(clean.groupby("rooms")["square_meters"].transform("mean"))


#%%
rentals = clean.query("to_rent == True")


#%%
import statsmodels.formula.api as smf

model = smf.ols("np.log(price + 1) ~ C(object_type) + square_meters + C(zip_code)", data=rentals.dropna()).fit()
print(model.summary())

#%%
from lightgbm import LGBMRegressor
FEATURES = ["object_type", "square_meters", "zip_code", "private_offer", "rooms"]
model = LGBMRegressor()
model.fit(rentals[FEATURES], rentals["price"])

#%%
from lightgbm.plotting import plot_importance
plot_importance(model)

#%%
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, rentals[FEATURES], ["square_meters"])  # TODO: pdp looks wonky
