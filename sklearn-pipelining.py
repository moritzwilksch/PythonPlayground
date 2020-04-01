# %%
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from customFunctions.functionDefinitions import *
import pandas as pd
import seaborn as sns
import sklearn
import numpy as np

# %%
df = pd.read_csv("./data/titanic.csv")
df.info()

# %%

# Defining columns
category_columns = ["Survived", "Pclass", "Sex", "Embarked"]
useless_cols = ["PassengerId", "Ticket", "Cabin"]

# Pipeline
pl1 = Pipeline(steps=[
    ("1", FunctionTransformer(categorize, kw_args=dict(category_columns=category_columns))),
    ("2", FunctionTransformer(drop_useless_cols, kw_args=dict(useless_cols=useless_cols))),
    ("3", FunctionTransformer(age_imputer)),
    ("4", FunctionTransformer(log_transform, kw_args=dict(col_to_transform='Fare')))
])

# Fit-transform pipeline
df = pl1.fit_transform(df)

# %%
# Pipelining with pandas
(df.pipe(categorize, category_columns=category_columns)
 .pipe(drop_useless_cols, useless_cols=useless_cols)
 .pipe(age_imputer)
 .pipe(log_transform, col_to_transform="Fare"))

# %%
# Save file
df.to_csv("./data/preparedTitanic.csv")

# %%
# Writing a pandas-friendly StandardScaler
# (one that preserves column names and returns a DF instead of a np.array)
from sklearn.base import TransformerMixin


class DFStandardScaler(TransformerMixin):
    """A pandas friendl Standard Scale. Takes a DataFrame, standardizes it and returns the standardized DataFrame
    with preserved column names."""

    def __init__(self):
        self.mean = np.nan
        self.std = np.nan
        self.colnames = pd.Index([])

    def fit(self, X: pd.DataFrame, y=None) -> TransformerMixin:
        self.mean = X.mean()
        self.std = X.std(ddof=0)
        self.colnames = X.columns
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.X = (X - self.mean) / self.std
        self.X.columns = self.colnames
        return self.X


customss = DFStandardScaler()
test = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8], 'c': [0, 1, 5, 3]})
print(f"== Original DF==\n{test}")
print("== Scaled DF==")
print(customss.fit_transform(test))
