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
df = pd.read_csv("./data/preparedTitanic.csv")
df = df.drop("Unnamed: 0", axis=1)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(df.drop(["Survived", "Name"], axis=1), df["Survived"])

xtrain = pd.get_dummies(xtrain)
xtest = pd.get_dummies(xtest)

from sklearn.linear_model import LogisticRegression
m1 = LogisticRegression(C=0.1, penalty="elasticnet", solver="saga", l1_ratio=1, max_iter=5000)
#m1 = LogisticRegression(max_iter=1000)
m1.fit(xtrain, ytrain)
preds = m1.predict(xtest)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, preds))
print(m1.coef_)

from sklearn.ensemble import RandomForestClassifier
m2 = RandomForestClassifier(max_features=4)
