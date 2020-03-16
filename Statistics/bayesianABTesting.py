# https://medium.com/hockey-stick/tl-dr-bayesian-a-b-testing-with-python-c495d375db4d
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
# and beta = 42 (the num of failures) which equates to a prior of roughly 19% conversion
a_posterior = beta(a_successes + 8, a_failures + 42)
b_posterior = beta(b_successes + 8, b_failures + 42)

# Sample from posterior
n_trials = 10000
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
import scipy.stats as stats

#Actual probabilities
p_A = 0.15
p_B = 0.14

conversion_rates_A = []
conversion_rates_B = []

for _ in range(5000):
    #User Traffic
    n_users = 13500
    n_A = stats.binom.rvs(n=n_users, p=0.5, size=1)[0]
    n_B = n_users - n_A

    #Conversions
    conversions_A = stats.bernoulli.rvs(p_A, size=n_A)
    conversion_rates_A.append(sum(conversions_A)/n_A)
    conversions_B = stats.bernoulli.rvs(p_B, size=n_B)
    conversion_rates_B.append(sum(conversions_B)/n_B)

#print("creative A was observed {} times and led to {} conversions".format(n_A, sum(conversions_A)))
#print("creative B was observed {} times and led to {} conversions".format(n_B, sum(conversions_B)))

# %%
sns.distplot(conversion_rates_A, label="A")
sns.distplot(conversion_rates_B, label="B")
plt.legend()

# %% [markdown]
# ## Chance of being Better
# %%
df = pd.DataFrame({"a": conversion_rates_A, "b": conversion_rates_B})
print(f"Chance of A being better than B = {np.sum(df.a > df.b)/len(df)}")
print(f"Chance of B being better than A = {np.sum(df.a < df.b)/len(df)}")