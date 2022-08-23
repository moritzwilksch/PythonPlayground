#%%
from audioop import lin2adpcm
from multiprocessing.connection import Pipe
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml


df = fetch_openml(data_id=41214, as_frame=True).frame
df

#%%
df = df.assign(Frequency=lambda d: d["ClaimNb"] / d["Exposure"])

#%%
plt.hist(df["Exposure"])

#%%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline

transformer = ColumnTransformer(
    [
        (
            "ohe",
            OneHotEncoder(sparse=False, drop="first"),
            ["Area", "VehGas", "Region", "VehBrand", "VehPower"],
        ),
        ("drop_useless", "drop", ["IDpol", "ClaimNb", "Frequency"]),
        (
            "scale_numeric",
            StandardScaler(),
            ["Exposure", "DrivAge", "VehAge", "BonusMalus", "Density"],
        ),
    ],
    remainder="passthrough",
)


X = transformer.fit_transform(df)
y = df["Frequency"]

#%%
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
from sklearn.ensemble import HistGradientBoostingRegressor


def evaluate(ytrue, preds):
    print("MAE = ", mean_absolute_error(ytrue, preds, sample_weight=df["Exposure"]))
    print("MSE = ", mean_squared_error(ytrue, preds, sample_weight=df["Exposure"]))
    print("MPD = ", mean_poisson_deviance(ytrue, preds, sample_weight=df["Exposure"]))


models = [
    LinearRegression(),
    PoissonRegressor(alpha=0),
    HistGradientBoostingRegressor(),
]

for model in models:
    print(f"{str(model.__class__):-^80}")
    model.fit(X, y, sample_weight=df["Exposure"])
    preds = model.predict(X)
    preds[preds < 0] = 1e-9
    evaluate(y, preds)

#%%
preds = model.predict(X)
preddf = pd.DataFrame(
    {
        "true": y,
        "pred": preds,
        "preds*exp": preds * df["Exposure"],
        "ClaimNb": df["ClaimNb"],
    }
)
preddf

#%%
preddf.assign(true=lambda d: d["true"].round()).groupby("true")["pred"].mean()

#%%

#%%
plt.scatter(y, preds)

#%%
df.query("Frequency == 0")

#%%
np.exp(X @ model.coef_ + model.intercept_)

#%%
model.predict(X)

#%%
import seaborn as sns

tips = sns.load_dataset("tips")
X = tips.drop("tip", axis=1)
y = tips["tip"]
y = np.log(y)
# X = X.assign(total_bill=np.log(X["total_bill"]))

transformer = ColumnTransformer(
    [("ohe", OneHotEncoder(), tips.select_dtypes("category").columns)],
    remainder="passthrough",
)

X = transformer.fit_transform(X)

# model = GammaRegressor()
model = LinearRegression()
model = HistGradientBoostingRegressor()

model.fit(X, y)
preds = model.predict(X)

print(mean_squared_error(y, preds))
print(mean_absolute_error(y, preds))

#%%
pd.DataFrame({"true": y, "pred": preds})

#%%
plt.scatter(y, preds - y)

#%%
fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
axes[0].scatter(y, preds)
axes[1].hist(y - preds)

#%%
from scipy import stats

x = np.arange(0, 10, 0.01)
y = stats.gamma(a=0.5, scale=1).pdf(x)
plt.plot(x, y)
