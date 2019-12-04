# %%
import pandas as pd
import seaborn as sns
import sklearn
import numpy as np

# %%
df = pd.read_csv("./data/titanic.csv")
df.info()

# %%
# FUNCTION DEFINITIONS
def categorize(df, category_columns=[]):
    """Changes data type of given columns to category"""
    try:
        df[category_columns] = df[category_columns].astype("category")
        df["Pclass"] = df["Pclass"].cat.reorder_categories([1,2,3])
    except:
        print("Categorizing failed!")
    finally:
        return df

def drop_useless_cols(df, useless_cols=[]):
    """Drops columns passed as useless_cols"""
    try:
        return df.drop(useless_cols, axis=1)
    except:
        print("No useless columns found!")
        return df

def age_imputer(df):
    """Imputes age column by taking mean per group"""
    try:
        df["Age"] = df["Age"].fillna(df.groupby(["Pclass", "Sex"])["Age"].transform("mean"))
    except:
        print("Exception while age imputing!")
    finally:
        return df

def log_transform(df, col_to_transform=""):
    try:
        df[col_to_transform] = np.log(df[col_to_transform].replace(to_replace=0, value=df[col_to_transform].mean()))
    except:
        print("Exception while log-transforming!")
    finally:
        return df

    

# %%
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