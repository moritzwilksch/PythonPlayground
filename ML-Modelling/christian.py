#%%
from tkinter import ON
import pandas as pd
import numpy as np


#%%
df = pd.read_parquet("data/berlin.parquet")

#%%
df = df.drop(["created_at", "location", "title", "url"], axis=1)

#%%
df[["price", "square_meters"]] = df[["price", "square_meters"]] / 100.0

#%%

#%%
df = df.loc[df["rooms"].isin(["1", "2", "3", "4"])]
df["rooms"] = df["rooms"].cat.remove_unused_categories()

df = df.loc[df["to_rent"] == True]
df = df.loc[df["price"] < 10_000]

df = df.drop("to_rent", axis=1)

#%%
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

CAT_COLS = ["object_type", "rooms", "zip_code"]

preprocessor = ColumnTransformer(
    [
        ("ohe", OneHotEncoder(sparse=False), CAT_COLS),
        ("ohe", OneHotEncoder(sparse=False), CAT_COLS),
    ],
    remainder="passthrough",
)

preprocessor.fit(df)
X = pd.DataFrame(preprocessor.transform(df), columns=preprocessor.get_feature_names_out())

#%%
y = X["remainder__price"]
X = X.drop(["remainder__price"], axis=1)


#%%
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y)


#%%
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

# model = LinearRegression()
model = HistGradientBoostingRegressor()
model.fit(xtrain, ytrain)


#%%
preds = model.predict(X)

#%%
preddf = pd.DataFrame({"true": y, "predicted": preds})
preddf = pd.concat([df.reset_index(drop=True), preddf.reset_index(drop=True)], axis=1)
preddf.head(30)

#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error

test_preds = model.predict(xtest)
print(mean_absolute_error(ytest, test_preds))

#%%
train_preds = model.predict(xtrain)
print(mean_absolute_error(ytrain, train_preds))