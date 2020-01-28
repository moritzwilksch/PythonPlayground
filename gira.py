#%%
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
df = sns.load_dataset("tips")
xtrain, xtest, ytrain, ytest = train_test_split(df[["total_bill", "sex", "smoker", "day", "time", "size"]], df["tip"])
xtrain = pd.get_dummies(xtrain)
xtest = pd.get_dummies(xtest)
# %%

sns.set_context("talk")
sns.regplot(data=df, x="total_bill", y="tip", color="red")

from sklearn.linear_model import LinearRegression
# Schritt 1
model = LinearRegression()

# Schritt 2
model.fit(xtrain, ytrain)

# Schritt 3
model.predict(xtest)

# %%
from sklearn.tree import DecisionTreeClassifier

m = DecisionTreeClassifier(max_depth=5)
m.fit(xtrain, ytrain)

