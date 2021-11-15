# %%
import tensorflow as tf
from sklearn.cluster import KMeans
from base64 import decode
import pandas as pd
import seaborn as sns
df = sns.load_dataset('titanic')#.drop('who adult_male embark_town alive alone'.split(), axis=1)

df.deck.cat.add_categories('NA', inplace=True)
df['age'] = df['age'].fillna(df.age.mean())
df['embarked'] = df['embarked'].fillna('S')
df['deck'] = df['deck'].fillna('NA')

# %%

km = KMeans(n_clusters=3)
clusters = km.fit_predict(pd.get_dummies(df))

# %%
dummytized = pd.get_dummies(df)
df.groupby(clusters).mean()


# %%

input_ = tf.keras.Input(shape=(32,))
encoded = tf.keras.layers.Dense(units=2, activation='relu')(input_)
decoded = tf.keras.layers.Dense(units=32, activation='linear')(input_)

model = tf.keras.Model(input_, decoded)
encoder = tf.keras.Model(input_, encoded)

model.compile(tf.keras.optimizers.Adam(0.005), 'mse')
model.fit(dummytized.values.astype('float64'), dummytized.values.astype('float64'), epochs=150)

#%%
encs = encoder.predict(dummytized.values.astype('float64'))

#%%
km2 = KMeans(n_clusters=3)
clusters2 = km2.fit_predict(encs)

#%%
df.groupby(clusters2).agg(('mean', 'size'))

#%%
import matplotlib.pyplot as plt
for col in df.columns:
    sns.scatterplot(x=encs[:, 0], y=encs[:, 1], hue=df[col])
    plt.show()