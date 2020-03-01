# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("./data/WIKI-AAPL.csv").drop(
    ["Ex-Dividend", "Open", "High", "Low", "Close", "Split Ratio", "Volume"], axis=1)
df["Date"] = df["Date"].astype("datetime64")
df.columns = ["date", "open", "high", "low", "close", "volume"]
df[["open", "high", "low", "close"]] = df[[
    "open", "high", "low", "close"]].astype("float16")
df["volume"] = df["volume"].astype("int32")
df = df.loc[df.date.dt.year >= 2005]
df.head()
df.info()

# %%
sns.set_style("ticks", rc={"figure.frameon": False, "font.family": "serif",
                           "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 0.9})
sns.lineplot(data=df, x="date", y="close")
plt.savefig("apple.svg")

# %%
df = sns.load_dataset("titanic")
g = sns.barplot(data=df, x="pclass", y="fare", palette=[
                "orange", "white", "white"], linewidth=1, edgecolor="black", errwidth=0)
g.patches[0].set_hatch("//")
g.patches[1].set_hatch("xxx")
g.patches[2].set_hatch("oo")

# %%
sns.barplot(data=df, y="pclass", x="fare", hue="survived", hue_order=[
            1, 0], edgecolor="black", dodge=False, orient="h", errwidth=0)

# %%
# Non-sense example, but shows how to plot with vanilla matplotlib via pandas, NOT SEABORN!
sns.set_style("ticks")
pt = pd.pivot_table(df, index="pclass",  columns="survived", values="fare")
pt.plot(kind="barh", stacked=True, edgecolor="black", color=["0.8", "0.2"])
plt.gca().set(xlabel="FARE", ylabel="PASSENGER CLASS", title="Fare per Class")

# %%
sns.set_style("ticks")
df = sns.load_dataset("tips")
g = sns.catplot(data=df, row="day", x="size", y="tip", kind="point",
                height=1, aspect=3, markers=".", color="#004260")

# %%
# Similar, but hacky
pt = pd.pivot_table(data=df, index="size", columns="day", values="tip")
fig, ax = plt.subplots(4, 1, sharex=True, figsize=(3, 5))
for i in range(4):
    clist = ["0.8"]*4
    clist[i] = "#004260"
    pt.plot(kind="line", ax=ax[i], legend=False, color=clist, marker=".")
    ax[i].annotate(xy=(1, 4), s=pt.columns[i], fontweight="bold")
    ax[i].set_yticks([2,5])
    ax[i].set_yticklabels(["$2", "$5"])
fig.suptitle("Tip per Group Size")
ax[2].set_ylabel("Tip")
ax[3].set_xlabel("Group Size")
