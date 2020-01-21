# %%
import pandas as pd
import requests
from bs4 import BeautifulSoup

# %%
user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14'
headers = {'User-Agent': user_agent,
    'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'}
quotepage = "https://www.uni-potsdam.de"
page = requests.get(quotepage, headers=headers).content
soup = BeautifulSoup(page, "html.parser")
kacheln = soup.find_all("div", attrs={"class": "up-a-teaser-2-text"})
res = [x.find("h2") for x in kacheln]
[x.text for x in res]

# %%
bloomberg_url = "https://www.bloomberg.com/quote/SPX:IND"
page = requests.get(bloomberg_url, headers=headers).content
soup = BeautifulSoup(page, "html.parser")
soup