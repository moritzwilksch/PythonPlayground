#%%
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

distributions = [
    {
        "name": "gamma",
        "params": (2, 1),
        "obj": stats.gamma,
        "type": "continuous",
    },
    {
        "name": "exponential",
        "params": (2,),
        "obj": stats.expon,
        "type": "continuous",
    },
    {
        "name": "poisson",
        "params": (2,),
        "obj": stats.poisson,
        "type": "discrete",
    },
]

fig, axes = plt.subplots(ncols=1, nrows=len(distributions), figsize=(5, 10), sharex=True)
for ax, distribution in zip(axes, distributions):
    x = (
        np.arange(0, 10, 0.01)
        if distribution["type"] == "continuous"
        else np.arange(0, 10)
    )
    y = (
        distribution["obj"](*distribution["params"]).pdf(x)
        if distribution["type"] == "continuous"
        else distribution["obj"](*distribution["params"]).pmf(x)
    )

    if distribution["type"] == "continuous":
        ax.plot(x, y)
    else:
        ax.bar(x, y)
