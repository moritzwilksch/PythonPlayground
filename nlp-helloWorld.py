# %%
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
# nltk.download()

# %%
import requests
from bs4 import BeautifulSoup
user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14'
fakeHeader = {'User-Agent': user_agent,
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'}
url = "https://www.uni-potsdam.de/de/social-media-krasnova.html"
raw = requests.get(url, fakeHeader).content
soup = BeautifulSoup(raw, "html.parser")
text = [x.text.replace("\n", "").strip().lower() for x in soup.find_all("p")][3:-11]

# %%
import twitter
api = twitter.Api(consumer_key = "RPy7jKwU9AA4IB631zs2bRiOM",
                  consumer_secret = "pqGdHgy41LWgIeVIuFeKTLglGmAuVC9lOCYeHeXjac1FA65i7O",
                  access_token_key = "4697971562-AdnzSxkVt61lBU7zs40CQklxa6ECwkrIpdBUlLl",
                  access_token_secret = "0E2djEwDxlxGK20aUd5yT0iTJpz2slKIKoon45kN3kTvW"
                  )
import pandas as pd
raw = api.GetUserTimeline(screen_name="NorbertGronau", count=200, include_rts=False)
from pandas.io.json import json_normalize
import re

df = json_normalize(raw[0].AsDict())
for r in raw:
    df = df.append(json_normalize(r.AsDict()))
df = df[1:].reset_index()
df["created_at"] = df["created_at"].astype("datetime64")
df["date"] = df["created_at"].dt.date
text = [r.lower().strip() for r in df.text]
text = [re.sub("http.*", "", t) for t in text]
pattern = "&amp;|\n|@\w*:?|[^\w\d\s]"
text = [re.sub(pattern, "", x) for x in text]

# %%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
sw = stopwords.words("german") + stopwords.words("english")
words = []
for l in text:
    words += word_tokenize(l)
words = [w for w in words if w not in sw and w not in string.punctuation]
words
nltk.FreqDist(words).plot(20)

# %%
from nltk.stem import WordNetLemmatizer

lmt = WordNetLemmatizer()
words = [lmt.lemmatize(w) for w in words]
words
nltk.FreqDist(words).plot(20)

# %%
df = pd.DataFrame(nltk.FreqDist(words).items(), columns=["word", "freq"])
df = df.sort_values(by="freq", ascending=False)
sns.barplot(data=df.iloc[0:20,:], x="word", y="freq", color="#004260")
plt.xticks(rotation=90)
