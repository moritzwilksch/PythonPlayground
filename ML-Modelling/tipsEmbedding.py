
# %%
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(1234)

# %% [markdown]
# # First Model: NN with OHE
# ## Data Preparation
# %%

df = sns.load_dataset("tips")


def prep_df(df):
    return df.pipe(
        pd.get_dummies
    ).pipe(
        StandardScaler().fit_transform
    )


xtrain, xtest, ytrain, ytest = train_test_split(
    df.drop("tip", axis=1), df.tip, test_size=0.2, random_state=1234)

xtrain = prep_df(xtrain)
xtest = prep_df(xtest)

# %% [markdown]
# ## Model Building
# ### Linear Regression (regulized)
# %%
linear_regression = ElasticNetCV()
linear_regression.fit(xtrain, ytrain)
pred = linear_regression.predict(xtest)
print(mean_absolute_error(ytest, pred))

# %%
# from tensorflow.python import keras # (autocomplete sucks)
base_nn = keras.models.Sequential([
    keras.layers.Dense(units=4, input_dim=12),
    keras.layers.Dense(units=1, activation="relu")
])

# %% [markdown]
# ## Model Fitting
base_nn.compile(optimizer="nadam", loss="mae", metrics=["mean_absolute_error"])
history = base_nn.fit(
    xtrain, ytrain, validation_data=(xtest, ytest), epochs=200)

# %%
lc = pd.DataFrame({"train_mae": history.history["mean_absolute_error"],
                   "val_mae": history.history["val_mean_absolute_error"]})
lc.plot()

# =================================================================================

# %% [markdown]
# # Second Model: NN with Embedding for Categoricals
# *** For using an extra branch in the NN (for embedding the days), one MUST USE THE FUNCTIONAL API!!!! ***

# %%
df = sns.load_dataset("tips")


def prep_df_embedding(df):
    """ Preps data. SEPEARATES DAY SERIES (must be int 0-3) for extra input into NN """
    day_series = LabelEncoder().fit_transform(pd.Series(df.day))
    df = pd.get_dummies(df.drop("day", axis=1)).pipe(
        StandardScaler().fit_transform
    )
    return df, day_series


xtrain, xtest, ytrain, ytest = train_test_split(
    df.drop("tip", axis=1), df.tip, test_size=0.2, random_state=1234)
xtrain, xtrain_days = prep_df_embedding(xtrain)
xtest, xtest_days = prep_df_embedding(xtest)

ytrain, ytest = np.array(ytrain), np.array(ytest)


# %%
# you MUST use the functional API to create models with branches!

# The input layers. Need to be specified explicitly
xin_day = keras.layers.Input(shape=(1,), name="day_data")
xin_rest = keras.layers.Input(shape=(8,), name="rest_data")

# Embedding the day-category. Concatenating the other (numerical) variables for the rest of the net
# Note the layer_object = instanciate(args)(input_to_layer) notation!!!
day_embedding = keras.layers.Embedding(4, 2, input_length=1)(xin_day)
day_embedding_reshaped = keras.layers.Reshape((2,))(day_embedding)
concat = keras.layers.Concatenate()([day_embedding_reshaped, xin_rest])

# The rest of the network
dense_layer_1 = keras.layers.Dense(units=4)(concat)
prediction = keras.layers.Dense(units=1, activation="relu")(dense_layer_1)

# No sequential model! Specify which objects are input and output
branched_model = keras.models.Model(
    inputs=[xin_day, xin_rest], outputs=prediction)

# Compile and fit.
# The input – instead of xtrain – is actually a list/dict of TWO arrays (1 for the days to be embedded, 1 for the rest)
branched_model.compile(optimizer="nadam", loss="mae",
                       metrics=["mean_absolute_error"])
history = branched_model.fit(x={"day_data": xtrain_days, "rest_data": xtrain},
                             y=ytrain, validation_data=([xtest_days, xtest], ytest), epochs=100)

# %%
# Plot learning curves
lc = pd.DataFrame({"train_mae": history.history["mean_absolute_error"],
                   "val_mae": history.history["val_mean_absolute_error"]})
lc.plot()

# %%
# Just for fun...
embeddings = pd.DataFrame(branched_model.layers[1].get_weights()[0], index=[
                          "Fri", "Sat", "Sun", "Thur"], columns=["x", "y"])
embeddings.plot(kind="scatter", x="x", y="y")

for name, a in embeddings.iterrows():
    plt.gca().annotate(name, (a[0], a[1]))
plt.show()

# %%
regression_df = df.copy()
regression_df[["emb_x", "emb_y"]] = regression_df.apply(lambda row: embeddings.loc[row["day"], :], axis=1)
regression_df = pd.get_dummies(regression_df.drop("day", axis=1))
linear_regression = ElasticNetCV()
xtrain, xtest, ytrain, ytest = train_test_split(regression_df.drop("tip", axis=1), regression_df.tip, test_size=0.2)
linear_regression.fit(xtrain, ytrain)
pred = linear_regression.predict(xtest)
print(mean_absolute_error(ytest, pred))