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
axes[1].hist()
