´
import pandas as pd

#%%
df = pd.read_csv("data/SRS_sentiment_labeled.csv")[["tweet", "sentiment"]]

#%%
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = TfidfVectorizer()
X = cv.fit_transform(df["tweet"])
y = df["sentiment"]

#%%
from sklearn.model_selection import train_test_split

xtrain, xval, ytrain, yval = train_test_split(X, y)

#%%
from sklearn.naive_bayes import GaussianNB


#%%
from sklearn.linear_model import LogisticRegression


model = LogisticRegression(max_iter=300, n_jobs=-1)
model.fit(xtrain, ytrain)

#%%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(yval, model.predict(xval)))
print(classification_report(yval, model.predict(xval)))
print(confusion_matrix(yval, model.predict(xval)))


#%%
s = "alsdfjkalösdfj"

neu = pd.DataFrame({"text": s})
cv.transform()