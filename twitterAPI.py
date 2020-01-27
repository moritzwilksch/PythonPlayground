
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
