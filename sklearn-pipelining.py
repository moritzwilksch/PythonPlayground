# %%
import pandas as pd
import seaborn as sns
import sklearn
import numpy as np

# %%
df = pd.read_csv("./data/titanic.csv")
df.info()

# %%
from customFunctions.functionDefinitions import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Defining columns
category_columns = ["Survived", "Pclass", "Sex", "Embarked"]
useless_cols = ["PassengerId", "Ticket", "Cabin"]

# Pipeline
pl1 = Pipeline(steps=[
    ("1", FunctionTransformer(categorize, validate=False, kw_args={"category_columns": category_columns})),
    ("2", FunctionTransformer(drop_useless_cols, validate=False, kw_args={"useless_cols": useless_cols})),
    ("3", FunctionTransformer(age_imputer, validate=False)),
    ("4", FunctionTransformer(log_transform, validate=False, kw_args={"col_to_transform": "Fare"}))
])

# Fit-transform pipeline
df = pl1.fit_transform(df)