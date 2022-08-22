#%%
from audioop import lin2adpcm
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
from sklearn.preprocessing import OneHotEncoder

transformer = ColumnTransformer(
    [
        ("ohe", OneHotEncoder(sparse=False), ["Area", "VehGas", "Region", "VehBrand"]),
        ("drop_useless", "drop", ["IDpol", "ClaimNb", "Exposure", "Frequency"]),
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
    print("MAE = ", mean_absolute_error(ytrue, preds))
    print("MSE = ", mean_squared_error(ytrue, preds))
    print("MPD = ", mean_poisson_deviance(ytrue, preds))


models = [
    LinearRegression(),
    PoissonRegressor(),
    HistGradientBoostingRegressor(loss="poisson"),
]

for model in models:
    print(f"{str(model.__class__):-^80}")
    model.fit(X, y)
    preds = model.predict(X)
    preds[preds < 0] = 1e-9
    evaluate(y, preds)

#%%
df["Frequency"].hist(log=True)

#%%
