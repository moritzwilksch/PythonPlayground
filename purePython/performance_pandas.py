# %%
import dask
import time
import re
import dask.dataframe as dd
import numpy as np
from pandas.core.dtypes.dtypes import str_type
from joblib import Parallel, delayed
from distributed import client
import pandas as pd

# %%
pddf = pd.read_parquet('~/Desktop/SMBA/data/jan_til_jun2019.parquet')

# %%

df: dd.core.DataFrame = dd.read_parquet('~/Desktop/SMBA/data/jan_til_jun2019.parquet')


# %%


def apply_fn(s: str):
    time.sleep(0.00001)
    return re.sub(r'(\$TSLA)', r'CASHTAG', s, re.IGNORECASE)


# %%
%%time
pddf.tweet.apply(apply_fn)

# %%
%%time
pddf.tweet.str.replace(r'(\$TSLA)', r'CASHTAG', regex=True)


# %%
%%time
df.tweet.apply(apply_fn, meta=('tweet', 'string')).compute()

# %%


# %%

# for tweet in df.tweet:
#     results.append(dask.delayed(counter.update)(tweet))

# %%

def second_apply_fn(df):
    time.sleep(0.00001)
    return df.tweet.str.replace(r'(\$TSLA)', r'CASHTAG', regex=True)


# %%
%%time
res = Parallel(n_jobs=8)(delayed(apply_fn)(tweet) for tweet in pddf.tweet)
pd.Series(res)

# %%
%%time
df.map_partitions(second_apply_fn).compute()


# %%
@dask.delayed
def call_api(x):
    time.sleep(0.5)
    return x


# %%
data = list(range(10))

# %%
lazy = [call_api(x) for x in data]
dask.compute(lazy)


#%%
import multiprocessing as mp

#%%
mp.set_start_method('fork')
with mp.Pool(8) as p:
    p.map(apply_fn, pddf.tweet.tolist())
