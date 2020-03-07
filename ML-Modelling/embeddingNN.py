# %%
import pandas as pd
import numpy as np
from tensorflow_core import keras

# %%
from sklearn.preprocessing import LabelEncoder

word_dict = {
    0: "hope",
    1: "to",
    2: "see",
    3: "you",
    4: "soon",
    5: "nice",
    6: "again"
}

X = np.array([
    [0, 1, 2, 3, 4],
    [5, 1, 2, 3, 6]
])

m = keras.Sequential([
    keras.layers.Embedding(7, 2)
])

m.compile("adam", "mse")
pred = m.predict(X)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

word_embedding = {w: m.predict([i])[0,0] for i, w in word_dict.items()}

df = pd.DataFrame(word_embedding, index=["x", "y"]).transpose()

df.plot(kind="scatter", x="x", y="y")

for w, a in word_embedding.items():
    plt.annotate(w, (a[0], a[1]+0.005))