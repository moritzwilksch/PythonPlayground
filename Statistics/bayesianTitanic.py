# %%
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = sns.load_dataset("titanic")

# %%
sns.countplot(data=df, x="sex", hue="survived")
pd.crosstab(columns=df.sex, index=df.survived)
# %%