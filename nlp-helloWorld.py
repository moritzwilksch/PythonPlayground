# %%
import pandas as pd
import numpy as np
import nltk
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


# %%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
sw = stopwords.words("english")
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


