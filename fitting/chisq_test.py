#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:21:14 2023

@author: fei
"""
import scipy.stats as stats
import numpy as np
from scipy.stats import chi2

def chisq_1d(expected,observed,rejection_factor):
    '''
    perform Chi-Square Goodness of Fit Test
        H0: (null hypothesis) A variable follows a hypothesized distribution.
        H1: (alternative hypothesis) A variable does not follow a hypothesized distribution.
    
        Chi-Square test statistic, and the corresponding p-value
        p correspond to dof = (len(datapoints)-1) = n-1
    '''
    statistic, pvalue = stats.chisquare(f_obs=observed, f_exp=expected)
    if pvalue > rejection_factor:
        return print('p-value=%.5f:'%pvalue,' fail to reject the null hypothesis')
    else:
        return print('p-value=%.5f:',pvalue,' reject the null hypothesis')

def chisq_ds(expected,observed,rejection_factor):
    '''
    perform Chi-Square Goodness of Fit Test
        H0: (null hypothesis) A variable follows a hypothesized distribution.
        H1: (alternative hypothesis) A variable does not follow a hypothesized distribution.
    
        Chi-Square test statistic, and the corresponding p-value
        p correspond to dof = (len(datapoints)-1) = n-1
    '''
    statistic, pvalue = stats.chisquare(f_obs=observed, f_exp=expected)
    chisq = np.sum(statistic)
    rows = observed.shape[0]
    cols = observed.shape[1]
    df = (rows - 1) * (cols - 1)
    p = 1 - chi2.cdf(chisq, df)
    if p > rejection_factor:
        print('p-value=%.5f:'%p,' fail to reject the null hypothesis')
    else:
        print('p-value=%.5f:'%p,' reject the null hypothesis')
#%% examples
#input data
expected1d = np.array([50, 50, 50, 50, 50])
observed1d = np.array([50, 60, 40, 47, 53])
rejection_factor1d = 0.05
chisq_1d(expected1d,observed1d,rejection_factor1d)

#input data
expected2d = np.array([[50, 50, 50, 50, 50],
                     [50, 50, 50, 50, 50],
                     [50, 50, 50, 50, 50]])
observed2d = np.array([[50, 60, 40, 47, 53],
                     [50, 66, 40, 47, 53],
                     [40, 60, 40, 47, 53]])
rejection_factor2d = 0.05
chisq_ds(expected2d,observed2d,rejection_factor2d)