#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from pdb import set_trace

#%%
df: pd.DataFrame = pd.read_csv('../data/telecom_churn.csv')
df.columns = [col.lower().replace(' ', '_') for col in df.columns]
df.info()

#%%
def eval_model(ytrue, preds):
    print(f"{'='*15} EVALUATI0N {'='*15}")
    print(f"ROC-AUC = {roc_auc_score(ytrue, preds)}")
    print("Confusion Matrix:")
    print(confusion_matrix(ytrue, preds))

#%%
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction import FeatureHasher

#%%
def ohe(df):
    df = df.copy()
    df = pd.get_dummies(df)
    return train_test_split(df.drop('churn', axis=1), df.churn, random_state=1234)

def drop_states(df):
    df = df.copy()
    df = pd.get_dummies(df.drop('state', axis=1))
    return train_test_split(df.drop('churn', axis=1), df.churn, random_state=1234)

def ordinal_encode(df):
    df = df.copy()
    oe = FeatureHasher(n_features=2)
    print(df.state.values.reshape(-1, 1))
    oe.transform(df.state.values.reshape(-1, 1))
    df[['state1', 'state2']] = oe.toarray()
    return train_test_split(df.drop('churn', axis=1), df.churn, random_state=1234)

#%%
## FIT AND EVALUATE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from typing import Callable
import keras

def fit_and_eval(prep_fun: Callable, df: pd.DataFrame):
    model1 = LogisticRegression(max_iter=300)
    xtrain, xtest, ytrain, ytest = prep_fun(df)
    model1.fit(xtrain, ytrain)
    eval_model(ytest, model1.predict(xtest))

    model2 = BernoulliNB()
    xtrain, xtest, ytrain, ytest = prep_fun(df)
    model2.fit(xtrain, ytrain)
    eval_model(ytest, model2.predict(xtest))

    model3 = RandomForestClassifier()
    xtrain, xtest, ytrain, ytest = prep_fun(df)
    model3.fit(xtrain, ytrain)
    eval_model(ytest, model3.predict(xtest))

#%%
fit_and_eval(ohe, df)

#%%
fit_and_eval(drop_states, df)

#%%
fit_and_eval(ordinal_encode, df)

#%%

xtrain, xtest, ytrain, ytest = ohe(df)

ytrain = ytrain.apply(int).values
ytest = ytest.apply(int).values

model4 = keras.Sequential([
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model4.compile('adam', 'binary_crossentropy')
model4.fit(xtrain.values, ytrain, validation_data=(xtest, ytest), epochs=100, batch_size=32)
eval_model(ytest, model4.predict_classes(xtest.values))

#%%
from sklearn.preprocessing import OrdinalEncoder
df['international_plan'] = df['international_plan'].map({'No':0, 'Yes':1})
df['voice_mail_plan'] = df['voice_mail_plan'].map({'No':0, 'Yes':1})
xtrain, xtest, ytrain, ytest = train_test_split(df.drop('churn', axis=1), df.churn)
xtrain_state = xtrain['state']
xtest_state = xtest['state']
oe = OrdinalEncoder()
xtrain_state = oe.fit_transform(xtrain_state.values.reshape(-1, 1))
xtest_state = oe.transform(xtest_state.values.reshape(-1, 1))
xtrain = xtrain.drop('state', axis=1)
xtest = xtest.drop('state', axis=1)

#%%
import keras
state_in = keras.Input(shape=(1,))
rest_in = keras.Input(shape=(18,))
emb = keras.layers.Embedding(51,5)(state_in)
emb_reshaped = keras.layers.Reshape((5,))(emb)
concat = keras.layers.Concatenate()([emb_reshaped, rest_in])
d1 = keras.layers.Dense(50, activation='relu')(concat)
d2 = keras.layers.Dense(25, activation='relu')(d1)
out = keras.layers.Dense(1, activation='sigmoid')(d2)
nn = keras.models.Model(inputs=[state_in, rest_in], outputs=out)
nn.compile('adam', 'binary_crossentropy')
nn.fit([xtrain_state, xtrain], ytrain, epochs=20, batch_size=32)

#%%
eval_model(ytest, nn.predict([xtest_state, xtest]) > 0.5)


#%%
from xgboost import XGBClassifier
from xgboost import plot_tree
df['state'] = df['state'].astype('category')
xtrain, xtest, ytrain, ytest = ohe(df)#train_test_split(df.drop('churn', axis=1), df.churn)
xgb = XGBClassifier()
#%%
xgb.fit(X=xtrain, y=ytrain, verbose=True)
eval_model(ytest, xgb.predict(xtest))

#%%
xgb.feature_importances_