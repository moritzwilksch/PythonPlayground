#%%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pandas as pd

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
    likelihood = pm.DiscreteUniform("lik", 1, n_lok, observed=OBSERVED)

    sampeled = pm.sample(chains=6)

az.plot_posterior(sampeled)


#%%
sampeled.to_dataframe()[("posterior", "n_lok")].mean()

#%%
sampeled.to_dataframe()
#%%
from scipy import stats

x = np.arange(0, 1000, 0.1)
y = stats.weibull_min(1, loc=60, scale=2).pdf(x)
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
with pm.Model() as m:
    sex = pm.Categorical("sex", [0.4, 0.6])

    if sex == 1:
        age_group = pm.Categorical("age_group", [0.4, 0.4, 0.2])
    else:
        age_group = pm.Categorical("age_group", [0.33, 0.33, 0.33])
    s = pm.sample(chains=1, draws=48)

#%%
data = s.to_dataframe()
pd.crosstab(data["sex"], data["age_group"])


#%%
@np.vectorize
def additional_profit_generated(price, cost, units_sold, inc_vol, pct_price_change):
    new_units_sold = units_sold * (1 + inc_vol * -pct_price_change / 0.1)
    print(f"{units_sold =}, {new_units_sold =}")

    old_profit = price - cost
    new_profit = price * (1 + pct_price_change) - cost
    print(f"{old_profit = }, {new_profit = }")
    return new_profit * new_units_sold - old_profit * units_sold


changes = np.arange(-0.1, 0.2, 0.01)
y = additional_profit_generated(
    price=110, cost=100, units_sold=50, inc_vol=0.2, pct_price_change=changes
)
plt.plot(changes, y)

#%%
import scipy
def criterion(delta):
    return -1 * additional_profit_generated(
        price=110, cost=100, units_sold=50, inc_vol=0.2, pct_price_change=delta
    )   

scipy.optimize.minimize(criterion, bounds=scipy.optimize.Bounds(-0.1, 0.2), x0=0)
