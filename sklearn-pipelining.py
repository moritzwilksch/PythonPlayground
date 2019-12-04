# %%
import pandas as pd
import sklearn

# %%
df = pd.read_csv("./data/titanic.csv")
df.info()

# %%
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

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Defining columns
category_columns = ["Survived", "Pclass", "Sex", "Embarked"]
useless_cols = ["PassengerId", "Ticket", "Cabin"]

# Pipeline
pl1 = Pipeline(steps=[
    ("1", FunctionTransformer(categorize, validate=False, kw_args={"category_columns": category_columns})),
    ("2", FunctionTransformer(drop_useless_cols, validate=False, kw_args={"useless_cols": useless_cols}))
])

# Fit-transform pipeline
df = pl1.fit_transform(df)