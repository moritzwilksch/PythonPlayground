# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("./data/WIKI-AAPL.csv").drop(["Ex-Dividend", "Open", "High", "Low", "Close", "Split Ratio", "Volume"], axis=1)
df["Date"] = df["Date"].astype("datetime64")
df.columns = ["date", "open", "high", "low", "close", "volume"]
df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype("float16")
df["volume"] = df["volume"].astype("int32")
df = df.loc[df.date.dt.year >= 2005]
df.head()
df.info()

# %%
sns.set_style("ticks", rc={"figure.frameon": False, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 0.9})
sns.lineplot(data=df, x="date", y="close")
plt.savefig("apple.svg")

# %%
df = sns.load_dataset("titanic")
g = sns.barplot(data=df, x="pclass", y="fare", palette= ["orange", "white", "white"], linewidth=1, edgecolor="black", errwidth=0)
g.patches[0].set_hatch("//")
g.patches[1].set_hatch("xxx")
g.patches[2].set_hatch("oo")

# %%
g = sns.barplot(data=df, x="pclass", y="fare", edgecolor="black")