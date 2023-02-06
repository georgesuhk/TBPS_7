#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:18:27 2023

@author: fei
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def pull_test(T,mu_T,T_sd,nbins,p0):
    pull = (mu_T-T)/T_sd
    n, bins, patches = plt.hist(pull, bins=nbins)    
    nsum=np.sum(n)
    nerr=((n/nsum)*(1-n/nsum))**0.5
    step = (max(bins)-min(bins))/(2*len(n))
    popt, pcov = curve_fit(gauss_function, bins[:len(n)]+step , n, p0 = p0, sigma=nerr)
    x = np.linspace(max(bins),min(bins),num=100)
    print(" Mu = %.2f +/- %.2f"%(popt[1],np.sqrt(pcov[1,1])))
    print(" Sig = %.2f +/- %.2f"%(popt[2],np.sqrt(pcov[2,2])))
    plt.axvline(popt[1],color = 'black',linestyle='--',label=" $\mu$ = %.2f +/- %.2f\n $\sigma$ = %.2f +/- %.2f"%(popt[1],np.sqrt(pcov[1,1]),popt[2],np.sqrt(pcov[2,2])))
    plt.plot(x, gauss_function(x, *popt))
    plt.errorbar(bins[:len(n)]+step, n, xerr=[0.5*step for i in range(len(nerr))], yerr=nerr,fmt='x',color='red',capsize=0)
    plt.xlabel('Pull')
    plt.ylabel('#')
    plt.title('Pull distribution')
    plt.legend()
    return plt.show()
#%% examples
#input data
import pandas as pd
##input: original SM data
expected = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                     50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                     50, 50, 50, 50, 50, 50, 50, 50, 50, 50])
##input: fitted values
observed = np.array([50, 60, 40, 47, 53, 50, 60, 40, 47, 53,
                     50, 66, 44, 45, 53, 50, 60, 40, 47, 53,
                     40, 60, 40, 48, 53, 50, 60, 40, 47, 53])
##input: fitted errors
observed_sd = np.array([5, 6, 3, 4, 3 ,1, 2, 3, 4, 5,
                     0, 6, 4, 7, 3, 1, 3 ,2, 2.5, 0.5,
                     4, 6, 4, 4, 3, 3, 4, 2, 1, 1])

nbins = 7
p0= [10, 0, 3]
pull_test(expected,observed,observed_sd,nbins,p0)
