#%%
import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

#%%
data = [0]* 64 + [1]*16
data_2 = [0]* 59 + [1]*21

with pm.Model() as model:
    ctr = pm.Beta("beta", alpha=1, beta=1)
    obs = pm.Binomial("binom", n=len(data), p=ctr, observed=sum(data))

    sampeled = pm.sample()
az.plot_trace(sampeled)


with pm.Model() as model:
    ctr = pm.Beta("beta", alpha=1, beta=1)
    obs = pm.Binomial("binom", n=len(data), p=ctr, observed=sum(data_2))

    sampeled_2 = pm.sample()
az.plot_trace(sampeled_2)

#%%
from scipy import stats as stats

x = np.arange(0.1, 0.4, 0.0001)
y = stats.beta(16, 64).pdf(x)
import matplotlib.pyplot as plt

plt.plot(x, y)

#%%
az.plot_posterior(sampeled)

#%%
(sampeled.to_dataframe()[('posterior', 'beta')] < sampeled_2.to_dataframe()[('posterior', 'beta')]).mean()
