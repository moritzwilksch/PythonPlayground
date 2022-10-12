#%%
import requests
from bs4 import BeautifulSoup

#%%
response = requests.get(
    "https://pola-rs.github.io/polars/py-polars/html/reference/expression.html"
)
soup = BeautifulSoup(response.content, "lxml")
#%%
all_links = soup.find_all("a", {"class": "reference internal"})

#%%
PAGE_PREFIX = "https://pola-rs.github.io/polars/py-polars/html/reference/"

direct_page_links = []
for link in all_links:
    try:
        direct_page_links.append(PAGE_PREFIX + link.attrs["href"])
    except:
        pass

#%%
has_no_example = []

for direct_link in direct_page_links:
    response = requests.get(direct_link)
    soup = BeautifulSoup(response.content, "lxml")
    highlights = soup.find_all("div", {"class": "highlight"})
    if not highlights:
        has_no_example.append(direct_link)


#%%
has_no_example = list(set(has_no_example))
with open("has_no_example.txt", "w") as f:
    f.writelines("\n".join(has_no_example))
    