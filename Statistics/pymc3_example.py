#%%
import pymc3 as pm
import numpy as np

#%%
import seaborn as sns
df = sns.load_dataset('tips')
a = df.tip[df.time=='Lunch'].values
b = df.tip[df.time=='Dinner'].values

#%%
sns.distplot(a)
sns.distplot(b)

#%%
with pm.Model() as model:
  avg_tip_a = pm.Uniform('avgtipa', 0, 5)
  avg_tip_b = pm.Uniform('avgtipb', 0, 5)

  delta = pm.Deterministic('delta', avg_tip_a - avg_tip_b)

  sigma_a = pm.Uniform('sigmaa', 0.5, 3)
  sigma_b = pm.Uniform('sigmab', 0.5, 3)

  tip_a = pm.Normal('realtipa', mu=avg_tip_a, sigma=sigma_a, observed=a)
  tip_b = pm.Normal('realtipb', mu=avg_tip_b, sigma=sigma_b, observed=b)

  trace = pm.sample(1000, tune=1000, cores=1)

#%%
import arviz as az
az.plot_trace(trace)

#%%
sns.distplot(trace['avgtipa'])
sns.distplot(trace['avgtipb'])

#%%
sns.distplot(trace['sigmaa'])
sns.distplot(trace['sigmab'])

#%%
(trace['delta'] > 0).sum()/len(trace['delta'])

#%%
resamples_a = np.random.choice(a, (1000, len(a)))
resamples_b = np.random.choice(b, (1000, len(b)))

diffs = np.mean(resamples_a, axis=1) - np.mean(resamples_b, axis=1)
sns.distplot(diffs)
print((diffs > 0).sum()/len(diffs))

#%%
a = np.array([0]*34 + [1]*17)
b = np.array([0]*21 + [1]*17)

with pm.Model() as model:
  # pa = pm.Uniform('pa')
  # pb = pm.Uniform('pb')

  pa = pm.Beta('pa', 17, 34)
  pb = pm.Beta('pb', 17, 21)

  delta = pm.Deterministic('delta', pa-pb)
  resa = pm.Binomial('resa', n=51, p=pa, observed=np.array([17]))
  resb = pm.Binomial('resb', n=38, p=pb, observed=np.array([17]))

  trace = pm.sample(1500, tune=1500, cores=1)

print(f"P(A > B) = {sum(trace['delta']>0)/len(trace['delta'])}")

#%%
import arviz as az
az.plot_trace(trace)

#%%
# RESAMPLING, equivalent to FLAT PRIOR
resamples_a = np.random.choice(a, (1500, 51))
resamples_b = np.random.choice(b, (1500, 38))

diffs = np.mean(resamples_a, axis=1) - np.mean(resamples_b, axis=1)
sns.distplot(diffs)

print(f"P(A > B) = {sum(diffs>0)/len(diffs)}")
