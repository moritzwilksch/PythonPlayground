# %%
import twitter
api = twitter.Api(consumer_key = "RPy7jKwU9AA4IB631zs2bRiOM",
                  consumer_secret = "pqGdHgy41LWgIeVIuFeKTLglGmAuVC9lOCYeHeXjac1FA65i7O",
                  access_token_key = "4697971562-AdnzSxkVt61lBU7zs40CQklxa6ECwkrIpdBUlLl",
                  access_token_secret = "0E2djEwDxlxGK20aUd5yT0iTJpz2slKIKoon45kN3kTvW"
                  )

# %%
import pandas as pd
raw = api.GetUserTimeline(user_id=50393960)
print(raw)

# %%
from pandas.io.json import json_normalize

df = json_normalize(raw[0].AsDict())
for r in raw:
    df = df.append(json_normalize(r.AsDict()))
df = df[1:].reset_index()
df["created_at"] = df["created_at"].astype("datetime64")
df["date"] = df["created_at"].dt.date

# %%
import seaborn as sns

# %% [markdown]
# # WITH TWEEPY

# %%
import createTwitterAPI, tweepy
api = createTwitterAPI.create()

# %%
# Works but only delivers 200 tweets max:
# raw = api.user_timeline("BillGates", count=1000, tweet_mode="extended")
status = [tweet for tweet in tweepy.Cursor(api.user_timeline, id="BillGates", tweet_mode="extended").items(500)]

# %%
