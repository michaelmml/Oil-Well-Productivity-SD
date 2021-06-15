import pandas as pd
import numpy as np
import scipy
import math
import csv
import random
import itertools
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Internal ipython tool for setting figure size
from IPython.core.pylabtools import figsize

# Seaborn for visualization
import seaborn as sns
sns.set_theme()

data = pd.read_csv('OGUS.csv')
data = data.drop(['API_No', 'Facility_ID', 'Qsec_location', 'Reported_Well_Status_A', 'Prod_Comments',
                  'Production_Confidential', 'Export Date'], axis=1)
data['Date_Production'] = pd.Series(pd.to_datetime(data['Date_Production'], format='%d/%m/%Y'))
data = data.dropna(subset=['Prod_Days'])
data['Oil_bbls'] = data['Oil_bbls'] + data['Total_Gas_mcf'] / 5.8
# data = data[data['Enhanced_Recovery_Unit'] == "None"]
data['Cumsum_Days'] = data.groupby(['Well_Name'])['Prod_Days'].apply(lambda x: x.fillna(0).cumsum())
data['Cumsum_OilProd'] = data.groupby(['Well_Name'])['Oil_bbls'].apply(lambda x: x.fillna(0).cumsum())

data.loc[data['Enhanced_Recovery_Unit'] != "None", 'Enhanced_Recovery_Unit'] = "Enhanced"
# print(data['Enhanced_Recovery_Unit'].describe())
data['Cumsum_OilProd_Norm'] = data[data['Enhanced_Recovery_Unit'] == "None"].groupby(['Well_Name'])['Oil_bbls'].apply(
    lambda x: x.fillna(0).cumsum())
data['Cumsum_OilProd_ERU'] = data[data['Enhanced_Recovery_Unit'] != "None"].groupby(['Well_Name'])['Oil_bbls'].apply(
    lambda x: x.fillna(0).cumsum())
data = data.fillna(0)

data = data.groupby(['Date_Production', 'Enhanced_Recovery_Unit'])['Oil_bbls'].sum().reset_index()
data = data.rename(columns={"Oil_bbls": "Oil Production"})
data = data.set_index('Date_Production')

y = data.resample('QS').sum()
print(y)
decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative', freq=40)
fig = decomposition.plot()
# plt.savefig('Decomposition.png')
plt.show()

"""
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 40) for x in list(itertools.product(p, d, q))]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}40 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
"""


mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 40),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
# plt.savefig('Diagnostics.png')
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2010':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Oil Production per Quarter')
plt.legend()
# plt.savefig('Predictions.png')
plt.show()