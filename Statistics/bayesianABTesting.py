# %%
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
a = stats.norm(0.15, 0.04)
b = stats.norm(0.19, 0.05)

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
b_successes = sum(b_group < 0.20)
a_failures = group_size - a_successes
b_failures = group_size - b_successes

# Model posterior as beta distribution
# Added to the prior beta parameters alpha = 8 (num of successes)
# and beta = 42 (the num of failures)
a_posterior = beta(a_successes + 8, a_failures + 42)
b_posterior = beta(b_successes + 8, b_failures + 42)

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

# %%
ct = np.array([[a_successes, a_failures], 
                [b_successes, b_failures]])
stats.chi2_contingency(ct)

# %%
means = [np.random.choice(b_samples, 500, replace=True).mean() for _ in range(1000)]
means = pd.Series(means).sort_values()
means.plot(kind="hist", bins=100)
np.percentile(means, [3, 97])
