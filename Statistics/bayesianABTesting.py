# %%
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
a = stats.norm(0.15, 0.04)
b = stats.norm(0.25, 0.05)

# %%
df = pd.DataFrame({"a": a.rvs(10000), "b": b.rvs(10000)})

# %%
df.plot(kind="hist", bins=100, alpha=0.6)

# %%
print(stats.ttest_ind(df.a, df.b))

# %% [markdown]
# # Bayesian A/B Testing
# %%
from scipy.stats import beta
np.random.seed(42)
group_size = 1000

a_group, b_group = np.random.rand(2, group_size)
# Set successrates for group A and B
# AND count number of successes/failures
a_successes = sum(a_group < 0.15)
b_successes = sum(b_group < 0.19)
a_failures = group_size - a_successes
b_failures = group_size - b_successes

# Model posterior as beta distribution
a_posterior = beta(a_successes, a_failures)
b_posterior = beta(b_successes, b_failures)

# Sample from posterior
n_trials = 100000
a_samples = pd.Series(a_posterior.rvs(n_trials))
b_samples = pd.Series(b_posterior.rvs(n_trials))
bwins = sum(b_samples > a_samples)

# %%
sns.distplot(a_samples)
sns.distplot(b_samples)
plt.show()

# %%
print(f"% of bwins = {bwins/n_trials}")
print(f"p-value = {stats.ttest_ind(a_samples, b_samples)[1]}")

# %% [markdown]
""" 
# Findings
p-value is already 0 for 15% VS. 16.2% success rates.
Percentage of bwin is > 0.95 only for 15% VS. 19% rates.
"""