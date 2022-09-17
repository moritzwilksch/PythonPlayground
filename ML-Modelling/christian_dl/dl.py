#%%
import re

import nltk
import pandas as pd
import polars as pl
import ray
import tensorflow as tf
from nltk.stem import PorterStemmer
from ray.util.multiprocessing import Pool

pool = Pool()

ps = PorterStemmer()
stopwords = nltk.corpus.stopwords.words("german")

#%%
# load data
df = pd.read_json(
    "/home/data-v3.json",
)


def prep_text(s: str) -> str:
    words = re.split(r"\s", s)
    words = [
        ps.stem(w)
        for w in words
        if w.lower() not in stopwords and len(w) > 2 and w != " "
    ]
    return " ".join(words)


preped_text = pool.map(prep_text, df["text"].to_list())


#%%
df_subset = df[["text", "relevant"]]
df_subset["text"] = preped_text

#%%
tok = tf.keras.preprocessing.text.Tokenizer()
tok.fit_on_texts(df_subset["text"])

#%%
sequence_of_ids = tok.texts_to_sequences(df_subset["text"])
sequence_of_ids_padded = tf.keras.utils.pad_sequences(sequence_of_ids, maxlen=128)

#%%
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(
    sequence_of_ids_padded, df_subset["relevant"], test_size=0.2, random_state=42
)


#%%
DROPOUT = 0.2
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(128,)),
        tf.keras.layers.Embedding(
            input_dim=len(tok.word_counts) + 1, output_dim=64
        ),  # 128 x 64
        # tf.keras.layers.Flatten(),  # 8192
        # tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.GRU(128, return_sequences=False),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

#%%
model.compile(
    "adam", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision()]
)
model.fit(x=xtrain, y=ytrain, validation_data=(xtest, ytest), epochs=1, batch_size=128)


#%%

#%%
from sklearn.metrics import classification_report

preds = model.predict(xtest)


#%%
print(classification_report(ytest, preds > 0.9))


#%%
s = """ 
text hier
 """
s = prep_text(s)
print(s)
s = tok.texts_to_sequences([s])
print(s)
s = tf.keras.utils.pad_sequences(s, maxlen=128)
print(s)
model.predict(s)

#%%
vectors = model.layers[0].weights[0]

word = "lose"
wordid = tok.texts_to_sequences([word])
vectors[wordid[0][0], :]

#%%
from sklearn.metrics.pairwise import cosine_similarity

top_similar = cosine_similarity(
    vectors[wordid[0][0], :].numpy().reshape(1, -1), vectors
).argsort()[0][::-1]

for ii in top_similar[:10]:
    print(tok.index_word[ii])
