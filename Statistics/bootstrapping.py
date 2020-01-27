# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
a = np.array([1,2,2,3,3,3,4,4,4,4,5,5,5,6,6,7])
sns.distplot(a)

# %%
a = np.random.normal(4, 2, 100)

n_straps = 10000
means = []
stds = []
for i in range(n_straps):
    c = np.random.choice(a, 50)
    means.append(c.mean())
    stds.append(c.std())

sns.distplot(means)
plt.show()
sns.distplot(stds)
plt.show()
# %%
import scipy.stats as st
st.t.interval(0.95, a)
