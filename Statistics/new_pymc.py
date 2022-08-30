#%%
from curses import nl
from email.errors import ObsoleteHeaderDefect
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

#%%
data = [0] * 64 + [1] * 16
data_2 = [0] * 59 + [1] * 21

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
(
    sampeled.to_dataframe()[("posterior", "beta")]
    < sampeled_2.to_dataframe()[("posterior", "beta")]
).mean()

#%%
# Locomotive Problem
from aesara import tensor as at


with pm.Model() as model:
    OBSERVED = 60
    n_lok = pm.Gamma("n_lok", alpha=1, beta=10)

    # likelihood = pm.Censored("censored_lik", pm.DiscreteUniform.dist(1, n_lok), lower=OBSERVED, upper=None)
    likelihood = pm.DiscreteUniform(
        "lik", 1, pm.Deterministic("n_lok/200", n_lok * 200 + 60), observed=OBSERVED
    )

    sampeled = pm.sample(chains=6)

az.plot_posterior(sampeled)


#%%
sampeled.to_dataframe()[("posterior", "n_lok")].mean()

#%%
sampeled.to_dataframe()
#%%
from scipy import stats

x = np.arange(0, 1000)
y = stats.expon().pdf((x - 60) / 200)
plt.plot(x, y)


#%%
y2 = stats.gamma(a=1).pdf(x / 200)
plt.plot(x, y2, color="red")

#%%
with pm.Model() as m:
    gamma = pm.Deterministic("n_lok/200", pm.Gamma("gamma", alpha=1, beta=1) * 200 + 60)
    res = pm.sample()
az.plot_posterior(res)

#%%
sampeled.__dir__()
