#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(font_scale=0.5)
pcts = np.array([61, 16, 14, 6, 3, 100]) *-1
bottoms = np.cumsum(([0] + pcts[:-2].tolist())).tolist() + [0]
plt.bar(x="Server Energy Cooling Networking Other Total".split(), height=pcts, bottom=bottoms, color="#0d0630 #18314F #384E77 #8BBEB2 #E6F9AF lightgrey".split(), ec=['k']*5 + ['None'])
for i, pct in enumerate(pcts):
    plt.annotate(f"{pct*-1}%", (i, bottoms[i] + 2), size=16, ha='center', annotation_clip=False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks(rotation=35)


#%%
ys = np.array([100, 20, -12, -50, 25, 83])
bots = np.cumsum([0] + ys[:-1].tolist())
bots = bots[:-1].tolist() + [0]
cols = ['red' if y < 0 else 'green' for y in ys.tolist()]
cols[-1] = "grey"
plt.bar(x=([1,2,3,4,5, 6]), height=ys, bottom=bots, color=cols)

bar_width = plt.gca().patches[0].get_width()

for i in range(4):
    print((bots+ys)[i])
    plt.hlines(y=(bots + ys)[i], xmin=i+1 - bar_width/2, xmax=i+2 + bar_width/2, color='k')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)