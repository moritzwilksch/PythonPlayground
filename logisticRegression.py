# %%
import pandas as pd
import seaborn as sns
import sklearn
import numpy as np

# %%
df = pd.read_csv("./data/preparedTitanic.csv")
df[["Sex", "Embarked", "Survived"]] = df[["Sex", "Embarked", "Survived"]].astype("category")
df.info()

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

xtrain, xtest, ytrain, ytest = train_test_split(df.drop(["Survived", "Name", "Unnamed: 0"], axis=1), df["Survived"])
xtrain = pd.get_dummies(xtrain)
xtest = pd.get_dummies(xtest)

# %%
m1 = LogisticRegression(max_iter=500)
m1.fit(xtrain, ytrain)
preds = m1.predict(xtest)
confusion_matrix(ytest, preds)

