# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:47:39 2019

@author: Xiao Liu
https://www.datahubbs.com/mcs_finance/
"""

#%%
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
#%%
def bsm_call(S_t, K, r, sigma, T):
    den = 1 / (sigma * np.sqrt(T))
    d1 = den * (np.log(S_t / K) + (r + 0.5 * sigma ** 2) * T)
    d2 = den * (np.log(S_t / K) + (r - 0.5 * sigma ** 2) * T)
    C = S_t * stats.norm.cdf(d1) \
        - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return C
#%%
start_date = '2015-01-01'
end_date = '2016-01-04'
fb_data = pdr.DataReader("FB", "yahoo", start_date, end_date)
fb_data.tail()
#%%
# Calculate volatility
# Returns
ret = fb_data['Close'][1:].values / fb_data['Close'][:-1].values - 1
# Volatility
sigma = np.std(ret) * np.sqrt(252)
print(sigma)
#%%
# Get the risk-free rate
rfr = pdr.DataReader("DGS10", "fred", end_date)
rfr.head()
#%%
# Get the opening price on the day of interest
S_t = fb_data['Open'][-1]
# Range of strike prices
K = S_t *(1 + np.linspace(0.05, 1, 20))
# Risk free rate on day of interest
r = rfr.loc[fb_data.index[-1]][0]
# Time to maturity in year fractions
T = 0.5

# Calculate option prices
C = [bsm_call(S_t, k, r / 100, sigma, T) for k in K]

plt.figure(figsize=(12,8))
plt.plot(K, C)
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.title("Option Price vs. Strike Price for 6-Month European Call Options")
plt.show()
#%%
# # Monte Carlo Option Pricing Model Implementation
# Keep values from previous BSM for comparison
K_bsm = K[0]
C_bsm = C[0]
#%%
np.random.seed(20)

# Parameters - same values as used in the example above 
# repeated here for a reminder, change as you like
# Initial asset price
S_0 = S_t
# Strike price for call option
K = K_bsm
# Time to maturity in years
T = 0.5
# Risk-free rate of interest
r = rfr.loc[fb_data.index[-1]][0] / 100
# Historical Volatility
sigma = np.std(ret) * np.sqrt(252)
# Number of time steps for simulation
n_steps = int(T * 252)
# Time interval
dt = T / n_steps
# Number of simulations
N = 100000
# Zero array to store values (often faster than appending)
S = np.zeros((n_steps, N))
S[0] = S_0

for t in range(1, n_steps):
    # Draw random values to simulate Brownian motion
    Z = np.random.standard_normal(N)
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * \
                             dt + (sigma * np.sqrt(dt) * Z))

# Sum and discount values
C = np.exp(-r * T) * 1 / N * np.sum(np.maximum(S[-1] - K, 0))
C

print("Strike price: {:.2f}".format(K_bsm))
print("BSM Option Value Estimate: {:.2f}".format(C_bsm))
print("Monte Carlo Option Value Estimate: {:.2f}".format(C))
#%%
plt.figure(figsize=(12,8))
plt.plot(S[:,0:10])
plt.axhline(K, c="k", xmin=0,
            xmax=n_steps,
           label="Strike Price")
plt.xlim([0, n_steps])
plt.ylabel("Non-Discounted Value")
plt.xlabel("Time step")
plt.title("First 10 Option Paths")
plt.legend(loc="best")
plt.show()
#%%









































