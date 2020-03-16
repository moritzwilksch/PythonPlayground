# SOURCE:
# https://spencerschien.info/post/interview_case_study/prompt/data-science-interview-case-study/

# %%
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
np.random.seed(123456789)
# %%


def make_truncnorm(lower=15, upper=25, mu=20, sigma=10/6):
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return stats.truncnorm(a, b, loc=mu, scale=sigma)


def get_num_corp_successes():
    _success_rate = 0.25
    return np.round(np.round(make_truncnorm().rvs(), 0) * _success_rate, 0)


def plot_dist(dist, xlim: Tuple[int, int] = (0, 100), disttype: str = "continuous"):
    xmin, xmax = xlim
    x = np.arange(xmin, xmax)
    if disttype == "continuous":
        y = dist.pdf(x)
        plt.plot(x, y)
        plt.show()
    else:
        y = dist.pmf(x)
        plt.bar(x, y)
        plt.show()


# %%

# STD of 100K is a guess as no real data is present
corp_donation = stats.lognorm(s=1, loc=np.mean([np.log(50000), np.log(1000000)]), scale=100_000)
plot_dist(corp_donation, (0, 1_000_000))

number_of_years = stats.poisson(mu=2)
plot_dist(number_of_years, (0, 10), disttype="discrete")

# %%
def generate_donation_timeline():
    donations = {}
    num_of_donors = get_num_corp_successes()
    for i in range(int(num_of_donors)):
        length = number_of_years.rvs()
        donations.update({i: length})
    
    longest_donation = max(donations.values())

    for donor in donations.keys():
        donations[donor] = [corp_donation.rvs()]*donations[donor]
        donations[donor] = np.pad(donations[donor], (0,longest_donation-len(donations[donor])))
    
    
    
    
    
    return donations

#%%
total_donations = []
for _ in range(10000):
    timeline = generate_donation_timeline()
    df = pd.DataFrame(timeline)
    cum_donations = np.cumsum(df.apply(sum, axis=1))
    total_donations.append(cum_donations.values[-1])
    # cum_donations.plot()
# plt.show()

# %%
sns.distplot(total_donations)
# CI95:
np.percentile(total_donations, (2.5, 97.5))