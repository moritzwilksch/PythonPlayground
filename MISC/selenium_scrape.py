#%%
import matplotlib.pyplot as plt
#%%
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument("--headless")

driver = webdriver.Chrome(options=chrome_options)
driver.get('https://www.instagram.com/explore/tags/id3/')

#%%
img = driver.find_elements_by_xpath("/html/body/div/section/main/article/div/div/div/div/div/a/div/div/img")

#%%
for i in img:
    a = plt.imread(i.get_attribute('src'), format='gif')
    plt.imshow(a)
    plt.show()