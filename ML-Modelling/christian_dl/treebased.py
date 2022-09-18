#%%
from pydoc import importfile
import re

import nltk
import pandas as pd
import polars as pl
import numpy as np
import ray
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

ray.init(ignore_reinit_error=True)
#%%
ps = PorterStemmer()
stopwords = nltk.corpus.stopwords.words("german")

# load data
df = pd.read_json(
    "/home/data-v3.json",
)


@ray.remote
def prep_text(s: str) -> str:
    words = re.split(r"\s", s)
    words = [
        ps.stem(w).lower()
        for w in words
        if w.lower() not in stopwords and len(w) > 2 and w != " "
    ]
    return " ".join(words)


prepped_text = [prep_text.remote(s) for s in df["text"]]
prepped_text = ray.get(prepped_text)

df["text"] = prepped_text
df_subset = df[["text", "relevant"]]

#%%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# vectorizer = TfidfVectorizer()
vectorizer = CountVectorizer(dtype=np.float32)
X = vectorizer.fit_transform(df_subset["text"])
y = df_subset["relevant"]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping, log_evaluation

model = LGBMClassifier(n_estimators=2000, random_state=42, importance_type="gain")
model.fit(
    xtrain,
    ytrain,
    eval_set=[(xtest, ytest)],
    eval_metric="average_precision",
    callbacks=[early_stopping(100, verbose=True), log_evaluation(period=50)],
)

#%%
from sklearn.metrics import classification_report

ypred = model.predict_proba(xtest)

#%%
print(classification_report(ytest, ypred[:, 1] > 0.8))

#%%
imps = pd.DataFrame(
    {"imp": model.feature_importances_, "name": vectorizer.get_feature_names_out()}
).sort_values("imp", ascending=False)

#%%
def classify(probas):
    return np.select(
        [
            probas[:, 1] > 0.8,
            probas[:, 1] < 0.2,
        ],
        choicelist=[1, 0],
        default=-1,
    )


predicted_classes = classify(ypred)

subset_with_prediction = predicted_classes[predicted_classes != -1]
print(f"- test set has {len(ytest)} samples")
print(
    f"- predicted 'UNCERTAIN' for {len(predicted_classes[predicted_classes == -1])} samples"
)
print(
    f"- predicted 'RELEVANT' for {len(subset_with_prediction[subset_with_prediction == 1])} samples"
)
print(
    f"- predicted 'NOT_RELEVANT' for {len(subset_with_prediction[subset_with_prediction == 0])} samples"
)

print(
    classification_report(ytest.values[predicted_classes != -1], subset_with_prediction)
)

#%%


def get_pct_uncertain(probas, margin=0.1):
    relevant_thresh = 1 - margin
    not_relevant_thresh = margin
    return ((probas < relevant_thresh) & (probas > not_relevant_thresh)).mean()


margins = np.arange(0.01, 0.5, 0.01)
pct_uncertain = [get_pct_uncertain(ypred[:, 1], margin=m) for m in margins]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(margins, pct_uncertain, color="blue")
ax.set(xlabel="margin", ylabel="% of samples predicted as uncertain")

ax.scatter(0.1, pct_uncertain[9], color="red", zorder=10)
ax.annotate(
    "margin = 0.1\n$\\rightarrow$ p < 0.1 = irrelevant & p > 0.9 = relevant\n$\\rightarrow$ 43% classified as uncertain",
    (0.11, pct_uncertain[9]),
)
ax.scatter(0.2, pct_uncertain[19], color="red", zorder=10)
ax.annotate(
    "margin = 0.2\n$\\rightarrow$ p < 0.2 = irrelevant & p > 0.8 = relevant\n$\\rightarrow$ 27% classified as uncertain",
    (0.21, pct_uncertain[19]),
)

sns.despine()
