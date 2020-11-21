#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_json("ScrapyTest/spiders/exportdata.json")

#%%
df['size'] = df['size'].astype("float")
#%%
plt.figure(figsize=(15, 7))
sns.scatterplot(data=df, x='size', y='preis', marker='o', edgecolor='k', color='blue')