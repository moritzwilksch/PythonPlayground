#%%
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import numpy as np
import matplotlib.pyplot as plt
#%%
import requests
import time
headers = {
  'Authorization': '88234shdfsdkl0_$1sdvRd01_233fdd',
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'
}

def scrape():
  all_dfs = []

  for start in range(0, 501, 10):
    try:
      url = f"https://de.indeed.com/jobs?q=Data+Scientist&start={start}"

      response = requests.request("GET", url, headers=headers)

      soup = BeautifulSoup(response.content, 'lxml')

      titles_raw = soup.find_all('a', attrs={'class': 'jobtitle turnstileLink'})
      companies_raw = soup.find_all('span', attrs={'class': 'company'})

      titles = [x['title'] for x in titles_raw]
      links = [f"http://www.indeed.com{x['href']}" for x in titles_raw]
      companies = [x.text.strip() for x in companies_raw]

      page_df = pd.DataFrame({
        'title': titles,
        'company': companies,
        'link': links
      })

      all_dfs.append(page_df)
      time.sleep(0.25)
    except Exception:
      print(f"skipping page {start}")

  df = pd.concat(all_dfs, axis=0, ignore_index=True).drop_duplicates(subset=['title', 'company'])
  return df

#%%
# df = scrape()
df = pd.read_csv('indeed_scrape.csv').drop('Unnamed: 0', axis=1)

#%%
skipped = []

def get_descr(url):
  time.sleep(np.random.randint(1, 4)/10.0)
  print(f"Getting URL {url[:80]}...")
  try:
    response = requests.request("GET", url, headers=headers)
    soup = BeautifulSoup(response.content, 'lxml')
    descr = soup.find_all('div', attrs={'class': 'jobsearch-JobComponent-description'})[0]
  except:
    print("Skipping {url}")
    skipped.append(url)
    return "SKIPPED"

  return descr.text

#%%
# descriptions = df.link.apply(get_descr)
# df['description'] = descriptions
# df.to_csv('indeed_scrape.csv')

#%%
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

eng_sw = stopwords.words('english')
de_sw = stopwords.words('german')

def clean_text(data, col='description'):
    stmr = PorterStemmer()

    # remove punctuation
    data[col] = data[col].str.replace(f"[{string.punctuation}]", "")
    
    # remove numbers
    data[col] = data[col].str.replace(r"\b\d+\b", "")

    # to lower
    data[col] = data[col].str.lower()

    # remove stopwords
    data[col] = data[col].apply(lambda twt: " ".join([word for word in twt.split() if word not in eng_sw + de_sw]))

    # stem words
    data[col] = data[col].apply(lambda twt: stmr.stem(twt))


    return data

clean = clean_text(df)
#%%
from nltk import FreqDist
plt.figure(figsize=(20, 10))
fd = FreqDist([word for x in clean.description.tolist() for word in x.split()])
fd.plot(50, cumulative=False)
plt.tight_layout()

#%%
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
counts = cv.fit_transform(clean.description)
# tf = TfidfVectorizer()
# counts = tf.fit_transform(clean.description)

#%%
from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(counts.toarray())
clean['agg_cluster'] = agg.labels_

#%%
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
km.fit(counts.toarray())
clean['km_cluster'] = km.predict(counts.toarray())

#%%
print(pd.crosstab(clean.km_cluster, clean.agg_cluster))

#%%
for i, txt in clean.groupby('km_cluster').sample(3).iterrows():
  print(txt['description'][:200])
  print("------")

#%%
print("AGG")

for i, txt in clean.groupby('agg_cluster').sample(3).iterrows():
  print(txt['description'][:200])
  print("------")

#%%
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=3)
lda.fit(counts.toarray())
# info loss!
clean['lda'] = np.argmax(lda.transform(counts), axis=1)

#%%
print("LDA")

for i, txt in clean.groupby('lda').sample(3).iterrows():
  print(txt['description'][:200])
  print("------")

v = {v: k for k, v in cv.vocabulary_.items()}

for topic in range(3):
  print(f"=== TOPIC #{topic} ===")
  for x in np.argsort(lda.components_, axis=1)[topic, -10:]:
      print(f"- {v[x]}")

#%%
# lda and agg are very similar
pd.crosstab(clean.lda, clean.agg_cluster)

#%%
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
for topic in range(3):
  xy_params = (counts, clean.agg_cluster==topic)
  lr.fit(*xy_params)
  print(f"=== TOP COEFS FOR TOPIC #{topic} ===")
  print(lr.score(*xy_params))
  for idx in np.argsort(lr.coef_)[0, -10:]:
    print(f"- {v[idx]}")
