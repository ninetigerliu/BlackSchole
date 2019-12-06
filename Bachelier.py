#!/usr/bin/env python
# coding: utf-8

# # Bachelier model

# The Bachier formula calculates the price of European put and call options. In this model, the price of the underlying asset follows a geometric Brownian motion:
# $$ dS = r dt + \sigma dW $$
# where
# * $S(t)$, the price of the underlying stock at time $t$
# * $r$, the constant interest rate.
# * $\sigma$, volatility. the stand deviative of the stock's returns.
# 
# This price is consistent with the PDE:
# $$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 \frac{\partial^2 V}{\partial S^2} + r S\frac{\partial V}{\partial S} - r V = 0 $$
# where
# * $V(S,t)$, the price of the option as a functio nof the underlying asset $S$ and $t$
#         
# The Value of a call(put) option for a underlying stock can be obtained by solving the equation for the corresponding terminal and boundary conditions: 
# \begin{align}
#      V(S, t) = e^{-rt}\big(\delta (F-K)\Phi(\delta d_1) + \sigma\sqrt{t} \phi(d_1) \big)
# \end{align}
# where
# * $\delta=\pm 1$ is call and put flag, respectively
# * $F=S e^{rt}$ is the forward price
# * $K$ is the strike
# * $ d_1 = \frac{1}{\sigma \sqrt{t}}(F-K)$
# * $ \Phi(x) = \frac{1}{\sqrt{2\pi}}\int_\infty^x e^{-t^2/2} dt $ is the cumulative distribution function of a standard normal distribution with zero mean and unit variance
# * $ \phi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} $ is the probability density function of a standard normal distribution with zero mean and unit variance

# In[1]:


import numpy as np
from scipy.stats import norm

#%% vannilla call and put
def Bachelier( CPFlag, s, k, t, sigma, r, isForwardPrice=False):
    
    f = s*np.exp(r*t)
    d1 = 1/np.sqrt(sigma*sigma*t)*(f-k)
    
    if isForwardPrice==False:
        df = np.exp(-r*t)
    else:
        df = 1.0
        
    res = {}
    phi = 1./np.sqrt(2*np.pi)*np.exp(-d1*d1/2)
    stdDev = sigma*np.sqrt(t)
    res['NPV'] = df*(CPFlag*(f-k)*norm.cdf(CPFlag*d1) + stdDev*phi)
    res['delta'] = CPFlag*df*norm.cdf(CPFlag*d1)
    res['gamma'] = df/stdDev*phi
    res['theta'] = -df * sigma * 0.5 / np.sqrt(t) * phi
    res['vega'] = df * phi * np.sqrt(t)
    return res;


# In[11]:


#%%
CPFlag = -1.0;
f = 10.0;
r = 0.03;
vol = 0.02;
t = 2;
k = 10.01;
s = f/np.exp(r*t);
res = Bachelier(CPFlag, s, k, t, vol, r);
#print ( npv );
#npv = Bachelier(CPFlag, s, k, t, vol, r, True);
print ( res );

