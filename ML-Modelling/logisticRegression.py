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

# %%
from tensorflow_core import keras

m2 = keras.Sequential([
    keras.layers.Dense(input_dim= 10, units=10, activation="tanh"),
    keras.layers.Dense(1, activation="sigmoid")
])

m2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])
m2.fit(np.array(xtrain), np.array(ytrain), epochs=50, validation_data=(np.array(xtest), np.array(ytest)))

# %%
preds = m2.predict_classes(np.array(xtest))
preds = preds.reshape(len(preds,))

confusion_matrix(ytest, preds)

wrong = df.loc[(ytest != preds).index]
df["classification_correct"] = np.ones((len(df), ))
df.loc[wrong.index, "classification_correct"] = 0
df["classification_correct"] = df["classification_correct"].astype("category")
# %%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

accuracy_score(ytest, preds)
confusion_matrix(ytest, preds)
roc_auc_score(ytest, preds)