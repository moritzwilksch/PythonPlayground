# %%
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %% [markdown]
# # Bayesian Updating of Beta Distribution
# %%
sns.set_style("ticks")
n_flips = [10, 100, 1000, 5000]
b = stats.bernoulli(0.5)


fig, ax = plt.subplots(len(n_flips), figsize=(5,5))
for i in range(len(n_flips)):
    results = b.rvs(n_flips[i])
    posterior = stats.beta(a=sum(results), b=len(results) - sum(results))
    x = np.arange(0, 1, 0.01)
    y = posterior.pdf(x)
    df = pd.DataFrame({"x": x, "y": y})
    sns.lineplot(data=df, x="x", y="y", ax=ax[i])
    ax[i].set_title(f"{n_flips[i]} trials")
    ax[i].set(xlabel="p")
    ax[i].vlines(x=0.5, ymin=df.y.min(), ymax=df.y.max(), colors="r")
sns.despine(top=True, right=True, left=False, bottom=False)
plt.subplots_adjust(hspace=3)
plt.show()

# %% [markdown]
# # Bootstrapping Samples from n Coinflips
# %%
n_bootstraps = 1000

n_flips = [10, 100, 1000, 5000]
b = stats.bernoulli(0.5)

fig, ax = plt.subplots(len(n_flips), figsize=(5,5), sharex=True)
for i in range(len(n_flips)):
    results = b.rvs(n_flips[i])
    bootstrapped_means = [np.random.choice(results, len(results)).mean() for _ in range(n_bootstraps)]
    sns.kdeplot(bootstrapped_means, ax=ax[i])
    ax[i].set_title(f"{n_flips[i]} trials")
    ax[i].set(xlabel="p")
sns.despine(top=True, right=True, left=False, bottom=False)
plt.subplots_adjust(hspace=3)
plt.show()