# %%
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import re
import tweepy
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
# nltk.download()

# %%
import requests
from bs4 import BeautifulSoup
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14'
fakeHeader = {'User-Agent': user_agent,
              'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
              'Accept-Encoding': 'none',
              'Accept-Language': 'en-US,en;q=0.8',
              'Connection': 'keep-alive'}
url = "https://www.uni-potsdam.de/de/social-media-krasnova.html"
raw = requests.get(url, fakeHeader).content
soup = BeautifulSoup(raw, "html.parser")
text = [x.text.replace("\n", "").strip().lower()
        for x in soup.find_all("p")][3:-11]

# %%
import createTwitterAPI, tweepy
api = createTwitterAPI.create()
# Works but only delivers 200 tweets max:
# raw = api.user_timeline("BillGates", count=1000, tweet_mode="extended")


def load_tweets_to_disk(username: str, n: int, filename: str):
    tweets = [tweet for tweet in tweepy.Cursor(
        api.user_timeline, id=username, tweet_mode="extended", include_rts=False).items(n)]
    with open(filename, "wb") as output_file:
        pickle.dump(tweets, output_file)


load_tweets_to_disk("keypousttchi", 200, "ML-Modelling/smallkp.pkl")

# %%
with open("ML-Modelling/pousttchiTweets.pkl", "rb") as input_file:
    tweets = pickle.load(input_file)

text = [s.full_text for s in tweets]
text = [r.lower().strip() for r in text]
text = [re.sub("http.*", "", t) for t in text]
pattern = "&amp;|\n|@\w*:?|[^\w\d\s]"
text = [re.sub(pattern, "", x) for x in text]
text

# %%
sw = stopwords.words("german") + stopwords.words("english")
words = []
for l in text:
    words += word_tokenize(l)
words = [w for w in words if w not in sw and w not in string.punctuation]
words
nltk.FreqDist(words).plot(20)

# %%

lmt = WordNetLemmatizer()
words = [lmt.lemmatize(w) for w in words]
words
nltk.FreqDist(words).plot(20)

# %%
df = pd.DataFrame(nltk.FreqDist(words).items(), columns=["word", "freq"])
df = df.sort_values(by="freq", ascending=False)
sns.barplot(data=df.iloc[0:20, :], x="word", y="freq", color="#004260")
plt.xticks(rotation=90)
