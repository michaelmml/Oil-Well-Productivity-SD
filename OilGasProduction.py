import pandas as pd
import numpy as np
import scipy
from scipy.stats import gamma
import math
import csv
import random
import matplotlib.pyplot as plt

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns

data = pd.read_csv('OGUS.csv')
data = data.drop(['API_No', 'Facility_ID', 'Qsec_location', 'Reported_Well_Status_A', 'Prod_Comments',
                  'Production_Confidential', 'Export Date'], axis=1)
data['Date_Production'] = pd.Series(pd.to_datetime(data['Date_Production'], format='%d/%m/%Y'))
data = data.dropna(subset=['Prod_Days'])
data['Oil_bbls'] = data['Oil_bbls'] + data['Total_Gas_mcf'] / 5800
data['Cumsum_Days'] = data.groupby(['Well_Name'])['Prod_Days'].apply(lambda x: x.fillna(0).cumsum())
data['Cumsum_OilProd'] = data.groupby(['Well_Name'])['Oil_bbls'].apply(lambda x: x.fillna(0).cumsum())

data.loc[data['Enhanced_Recovery_Unit'] != "None", 'Enhanced_Recovery_Unit'] = "Enhanced"
# print(data['Enhanced_Recovery_Unit'].describe())
data['Cumsum_OilProd_Norm'] = data[data['Enhanced_Recovery_Unit'] == "None"].groupby(['Well_Name'])['Oil_bbls'].apply(
    lambda x: x.fillna(0).cumsum())
data['Cumsum_OilProd_ERU'] = data[data['Enhanced_Recovery_Unit'] != "None"].groupby(['Well_Name'])['Oil_bbls'].apply(
    lambda x: x.fillna(0).cumsum())
data = data.fillna(0)

# Create database by well and production and no. of days characteristics of each well
allprodmaxindex = data.groupby(['Well_Name'])['Cumsum_OilProd'].idxmax()
allprodcum = allprodmaxindex.apply(lambda x: data['Cumsum_OilProd'].loc[x])
allprodquarter = allprodcum / 4
allprodhalf = allprodcum / 2
allprodthreequarter = allprodcum * (3 / 4)
allprodrate = pd.concat([allprodmaxindex, allprodcum, allprodquarter, allprodhalf, allprodthreequarter], axis=1)
allprodrate.reset_index(inplace=True)
allprodrate.columns = ['Well_Name', 'allprodmaxindex', 'allprodcum', 'allprodquarter', 'allprodhalf',
                       'allprodthreequarter']

# Find index at specific production levels
allprodquarterindex = allprodrate[['Well_Name', 'allprodquarter']].apply(lambda x: abs(data.loc[(data['Well_Name'] ==
                        x['Well_Name']), 'Cumsum_OilProd'] - x['allprodquarter']).idxmin(), axis=1)
allprodhalfindex = allprodrate[['Well_Name', 'allprodhalf']].apply(lambda x: abs(data.loc[(data['Well_Name'] ==
                        x['Well_Name']), 'Cumsum_OilProd'] - x['allprodhalf']).idxmin(), axis=1)
allprodthreequarterindex = allprodrate[['Well_Name', 'allprodthreequarter']].apply(
    lambda x: abs(data.loc[(data['Well_Name'] ==
                            x['Well_Name']), 'Cumsum_OilProd'] - x['allprodthreequarter']).idxmin(), axis=1)

# Use of index to find the number of days
allprodquarterdays = allprodquarterindex.apply(lambda x: data['Cumsum_Days'].loc[x])
allprodhalfdays = allprodhalfindex.apply(lambda x: data['Cumsum_Days'].loc[x])
allprodthreequarterdays = allprodthreequarterindex.apply(lambda x: data['Cumsum_Days'].loc[x])

allprodrate = pd.concat([allprodrate, allprodquarterdays, allprodhalfdays, allprodthreequarterdays], axis=1)
allprodrate = allprodrate.drop(['allprodquarter', 'allprodhalf', 'allprodthreequarter'], axis=1)
allprodrate.columns = ['Well_Name', 'allprodmaxindex', 'allprodcum', 'allprodquarter', 'allprodhalf',
                       'allprodthreequarter']
allprodrate['halfwaydays'] = allprodrate['allprodhalf'] - allprodrate['allprodquarter']
allprodrate['threequarterdays'] = allprodrate['allprodthreequarter'] - allprodrate['allprodhalf']
print(allprodrate.describe())

allprodrate['halfdaysdate'] = (allprodhalfindex.apply(lambda x: data['Date_Production'].loc[x])).dt.year

# Production amount by normal and enhanced recovery units
allprodmaxindexnorm = data.groupby(['Well_Name'])['Cumsum_OilProd_Norm'].idxmax()
allprodcumnorm = allprodmaxindexnorm.apply(lambda x: data['Cumsum_OilProd_Norm'].loc[x])
allprodmaxindexERU = data.groupby(['Well_Name'])['Cumsum_OilProd_ERU'].idxmax()
allprodcumERU = allprodmaxindexERU.apply(lambda x: data['Cumsum_OilProd_ERU'].loc[x])

temp = pd.concat([allprodcumnorm, allprodcumERU], axis=1)
temp.reset_index(inplace=True)
temp = temp.drop(['Well_Name'], axis=1)

allprodrate = pd.concat([allprodrate, temp], axis=1)
allprodrate = allprodrate[allprodrate['threequarterdays'] > 100]
allprodrate.to_csv('prodrate.csv', index=True)
allprodrate.info()

# -------------------------------- Charts and Analysis -----------------------------------------------

# Area chart over time of total production per month through normal and ERU
data = data.groupby(['Date_Production', 'Enhanced_Recovery_Unit'])['Oil_bbls'].sum().reset_index()
data = data.set_index('Date_Production')
# data['Oil_bbls'].resample('Y').sum()

data['Pre_Enhanced'] = data.loc[data['Enhanced_Recovery_Unit'] == "None", 'Oil_bbls']
data.loc[data['Enhanced_Recovery_Unit'] == "None", 'Oil_bbls'] = 0
data.plot.area()

# Productivity calculated by (3/4) of total cumulative number of barrels over the number of days to reach that level
allprodrate['overallrate'] = (allprodrate['allprodcum'] * (3 / 4)) / allprodrate['allprodthreequarter']

# Removing wells with productivity below 1 barrel per day
allprodrate = allprodrate[(allprodrate['overallrate'] > 1)]

# Histogram
figsize(8, 8)
plt.hist(allprodrate['overallrate'].dropna(), bins=50, edgecolor='black');
plt.xlabel('BOE Per Day');
plt.ylabel('Count');
plt.title('BOE Per Day');


# Monte Carlo Markov Chain with Metropolis-Hastings Algorithm
# Idea from histogram of productivity that the distribution approximates the Gamma Distribution - use of MCMC
# Logarithm for faster computation
def log_gamma(x, data):
    return np.sum(np.log(gamma(a=x[0], scale=x[1], loc=0).pdf(data)))


# Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        return (accept < (np.exp(x_new - x)))

# We need a proposal mechanism which will take us around the space
def proposal(x):
    if x[0] <= 0 or x[1] <= 0:
        return 0
    else:
        return 1


# Essentially, acceptance probability = min(1, dist(y)proposal(x|y)/dist(x)proposal(y|x))
# = min(1, exp(log(dist(y)proposal(x|y)/dist(x)proposal(y|x))
# = min(1, exp(   log(dist(y)) + log(proposal(x|y)) - log(dist(x)) + log(proposal(y|x))   )
# as given in MH function and Acceptance function
def metropolis_hastings(p, q, transition_dist, initial_parameters, iterations, data, accept_rule):
    x = initial_parameters
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new = transition_dist(x)
        if accept_rule(p(x, data) + np.log(q(x)), p(x_new, data) + np.log(q(x_new))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)

    return np.array(accepted), np.array(rejected)

# To move x[a,scale] by the normal distribution
transition_pdf = lambda x: np.random.normal(x, [0.1, 2], size = (2,))

accepted, rejected = metropolis_hastings(log_gamma, proposal, transition_pdf, [3, 10], 50000,
                                         allprodrate['overallrate'], acceptance)
print(accepted.shape)
print(accepted[-10:])

# Metropolis_Hastings burn-in and convergence illustration
fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(3, 1, 1)
ax.plot(accepted[:50, 0], accepted[:50, 1], label="Path")
ax.plot(accepted[:50, 0], accepted[:50, 1], 'b.', label='Accepted')
ax.plot(rejected[:50, 0], rejected[:50, 1], 'rx', label='Rejected')
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.legend()
ax.set_title("Figure: MCMC sampling for $a$ and $b$ with Metropolis-Hastings. First 50 samples are shown.")

ax = fig.add_subplot(3, 1, 2)
ax.plot(accepted[:, 0], accepted[:, 1], label="Path")
ax.plot(accepted[:, 0], accepted[:, 1], 'b.', label='Accepted', alpha=0.3)
ax.plot(rejected[:, 0], rejected[:, 1], 'rx', label='Rejected', alpha=0.3)
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.legend()
ax.set_title("Figure: MCMC sampling for $a$ and $b$ with Metropolis-Hastings. All samples are shown.")

to_show = 50
ax = fig.add_subplot(3, 1, 3)
ax.plot(accepted[-to_show:, 0], accepted[-to_show:, 1], label="Path")
ax.plot(accepted[-to_show:, 0], accepted[-to_show:, 1], 'b.', label='Accepted', alpha=0.5)
ax.plot(rejected[-to_show:, 0], rejected[-to_show:, 1], 'rx', label='Rejected', alpha=0.5)
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.legend()
ax.set_title("Figure: MCMC sampling for $a$ and $b$ with Metropolis-Hastings. Last 50 samples are shown.")

show = -int(0.5 * accepted.shape[0])

mu = accepted[show:, 0].mean()
sigma = accepted[show:, 1].mean()
print(mu, sigma)

model = lambda t, mu, sigma: np.random.gamma(mu, sigma, t)
t = np.arange(allprodrate['overallrate'].shape[0])
observations = model(t.shape[0], mu, sigma)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)

# Histogram of observed productivity and predicted productivity based on Gamma Distribution generated by
# Metropolis-Hastings

ax.hist(observations, histtype='stepfilled', density=True, edgecolor='none', bins=30, label="Predicted values")
ax.hist(allprodrate['overallrate'], histtype='stepfilled', density=True, edgecolor='none', bins=30, alpha=0.5, label="Original values")
ax.set_xlabel("Productivity - BoE / Day")
ax.set_ylabel("Density")
ax.set_title("Observed vs Predicted: Productivity by Well")
ax.legend()