#%%
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#%%
df = sns.load_dataset("tips")

#%%

model = smf.ols(
    formula="tip ~ np.log(total_bill) + sex + smoker + day + time + size", data=df
)
results = model.fit()
results.summary()

#%%
residuals = results.predict(df) - df["tip"]

#%%
plt.hist(np.log(df["tip"]))

#%%
from scipy import stats

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
rvs = stats.expon(scale=1).rvs(size=1000)
axes[0].hist(rvs)


#%%
a = stats.norm(loc=0.2, scale=0.01).rvs(size=1000)
b = stats.norm(loc=0.24, scale=0.02).rvs(size=1000)
fig, ax = plt.subplots()
sns.kdeplot(a, shade=True, ax=ax)
sns.kdeplot(b, shade=True, ax=ax)

#%%
deltas = a - b
sns.kdeplot(np.random.choice(deltas, size=(2000, 1000)).mean(axis=1))

#%%
new_a = np.random.choice(a, size=(2000, 1000)).mean(axis=1)
new_b = np.random.choice(b, size=(2000, 1000)).mean(axis=1)
deltas = new_a - new_b
sns.kdeplot(deltas)