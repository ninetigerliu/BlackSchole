#!/usr/bin/env python
# coding: utf-8

# # BlackSchole Formulas

# The Black–Scholes formula calculates the price of European put and call options. In this model, the price of the underlying asset follows a geometric Brownian motion:
# $$ \frac{dS}{S} = \mu dt + \sigma dW $$
# where
# * $S(t)$, the price of the underlying stock at time $t$
# * $\mu$, the drift.
# * $\sigma$, volatility. the stand deviative of the stock's returns.
# 
# This price is consistent with the Black–Scholes equation:
# $$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q-b) S\frac{\partial V}{\partial S} - r V = 0 $$
# where
# * $V(S,t)$, the price of the option as a functio nof the underlying asset $S$ and $t$
# * $r, q$ and $b$, the risk free interest rate, continuous divident rate and borrow rate, respectively.
#         
# The Value of a call(put) option for a underlying stock can be obtained by solving the equation for the corresponding terminal and boundary conditions: 
# \begin{align}
#      C(S, t) = &amp;N(d_1)S e^{-(q+b)t} - N(d_2)Ke^{-r t} \\
#      P(S, t) = &amp;N(-d_2)Ke^{-rt} - N(-d_1)S  e^{-(q+b)t}  
# \end{align}
# where
# * $S$ is the stock price at time 0
# * $K$ is the strike
# * $ d_1 = \frac{\ln(\frac{S}{K}) + (r-q-b + \frac{1}{2} \sigma^2 )t}{\sigma \sqrt{t}}$
# * $ d_2 = d_1 -\sigma \sqrt{t} $

# In[40]:


import numpy as np
from scipy.stats import norm

#%% vannilla call and put
def BS( CPFlag, s, k, t, sigma, r, q, b):

    stdDev = sigma*np.sqrt(t)
    d1 = (np.log(s/k) + ( r - q - b + 0.5*sigma*sigma)*t)/stdDev
    d2 = d1 - stdDev  
    
    df = np.exp(-r*t)
    dfCost = np.exp(-(q+b)*t)
    cdfD1 = norm.cdf(CPFlag*d1)
    cdfD2 = norm.cdf(CPFlag*d2)
    #print('cdfD1='+str(format(cdfD1,'.15f')))
    phi = norm.pdf(d1)
    res = {}
    res['NPV'] = CPFlag*(s*dfCost*cdfD1 - k*df*cdfD2)
    res['delta'] = CPFlag*dfCost*cdfD1
    res['gamma'] = dfCost*phi/s/stdDev
    res['theta'] = -dfCost*s*phi*sigma/2/np.sqrt(t) -CPFlag*r*k*df*norm.cdf(CPFlag*d2) + CPFlag*(q+b)*s*dfCost*norm.cdf(CPFlag*d1)
    res['vega'] = dfCost*s*phi*np.sqrt(t)
    return res;


# In[49]:


#%%
CPFlag = -1.0;
s = 100.0;
k = 90.0;
r = 0.05;
q = 0.01;
b = 0.0;
t = 1.;  #in years
vol = 0.25;
df = np.exp(-r*t)
f = s/df;
npv = BS(CPFlag, s, k, t, vol, r, q, b);
ds = 1e-4
npv1 = BS(CPFlag, s+ds, k, t, vol, r, q, b)
npv2 = BS(CPFlag, s-ds, k, t, vol, r, q, b)
dt = 1e-6
npvdt = BS(CPFlag, s, k, t+dt, vol, r, q, b)
dvol = 1e-6
npvdvol = BS(CPFlag, s, k, t, vol+dvol, r, q, b)
print("gamma new = " + str((npv1['NPV'] - 2*npv['NPV']+npv2['NPV'])/ds/ds ))
print("vega new = " + str((npvdvol['NPV'] - npv['NPV'])/dt ))
print("theta new = " + str(-(npvdt['NPV'] - npv['NPV'])/dt ))
print("delta new = " + str((npv1['NPV'] - npv2['NPV'])/ds/2 ))
print ( npv );

