# %%
import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
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

    
