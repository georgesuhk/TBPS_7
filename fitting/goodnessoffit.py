#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:18:27 2023

@author: fei


"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.stats import chi2

def chisq(T,mu_T,rejection_factor):
    '''
    perform Chi-Square Goodness of Fit Test
        H0: (null hypothesis) A variable follows a hypothesized distribution.
        H1: (alternative hypothesis) A variable does not follow a hypothesized distribution.
    
        Chi-Square test statistic, and the corresponding p-value
        p correspond to dof = (len(datapoints)-1) = n-1
    '''
    statistic, pvalue = stats.chisquare(f_obs=mu_T, f_exp=T)
    if pvalue > rejection_factor:
        return print('p-value=%.5f:'%pvalue,' fail to reject the null hypothesis')
    else:
        return print('p-value=%.5f:',pvalue,' reject the null hypothesis')

#%% Thesis method: 
'''
A pull-study is conducted to test the correction of the veto.

    T is the value returned by the fit, (It is the array of fitted value of bin height.)
    mu_T is the value with which the toy sample was generated, (It is the array of bin height of distribution of the parameter.)
    and T_sd is the uncertainty returned by the fit. (It is the uncertainty of paramater after fitting.)

'pull_test' gives the (gaussian) distribution
By generating many event samples, fitting each one and calculating the pull 
for every parameter in every sample it can be tested whether the fit returns 
the correct parameters without any biases.
'''
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def pull_test(T,mu_T,T_sd,nbins=10,p0):
    pull = (mu_T-T)/T_sd
    n, bins = np.histogram(pull, bins=nbins,weights = None) 
    
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})  
    nsum=np.sum(n)
    nerr=((n/nsum)*(1-n/nsum))**0.5
    step = (max(bins)-min(bins))/(2*len(n))
    popt, pcov = curve_fit(gauss_function, bins[:len(n)]+step , n, p0 = p0, sigma=nerr)
    x = np.linspace(max(bins),min(bins),num=100)
    print(" Mu = %.2f +/- %.2f"%(popt[1],np.sqrt(pcov[1,1])))
    print(" Sig = %.2f +/- %.2f"%(popt[2],np.sqrt(pcov[2,2])))
    a0.plot(x, gauss_function(x, *popt))
    a0.hist(pull, bins=nbins,color='xkcd:powder blue')
    a0.axvline(popt[1],color = 'black',linestyle='--',label=" $\mu$ = %.2f +/- %.2f\n $\sigma$ = %.2f +/- %.2f"%(popt[1],np.sqrt(pcov[1,1]),popt[2],np.sqrt(pcov[2,2])))
    a0.errorbar(bins[:len(n)]+step, n, xerr=[0.5*step for i in range(len(nerr))], yerr=nerr,fmt='x',color='red',capsize=0)
    a0.set_xlabel('Pull')
    a0.set_ylabel('Number of events')
    a0.set_title('Pull distribution')
    a0.legend()
    a1.plot(T,pull,'.')
    a1.set_xlabel('T')
    a1.set_ylabel('Pull')
    return f.tight_layout()

######################################################################
#%%Jean's method
import numpy as np

def pull_test2(T,mu_T,distance):
    d = (mu_T-T)
    count=0
    for i in range (0,len(d)):
        if np.absolute(d[i])<distance:
            count+=1
    return print('Percentage rate of passing pull test=',count/len(d))
    
