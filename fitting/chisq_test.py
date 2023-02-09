#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:21:14 2023

@author: fei
"""
import scipy.stats as stats
import numpy as np
from scipy.stats import chi2

def chisq(expected,observed,rejection_factor):
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

# def chisq_ds(expected,observed,rejection_factor):
#     '''
#     perform Chi-Square Goodness of Fit Test
#         H0: (null hypothesis) A variable follows a hypothesized distribution.
#         H1: (alternative hypothesis) A variable does not follow a hypothesized distribution.
    
#         Chi-Square test statistic, and the corresponding p-value
#         p correspond to dof = (len(datapoints)-1) = n-1
#     '''
#     statistic, pvalue = stats.chisquare(f_obs=observed, f_exp=expected)
#     chisq = np.sum(statistic)
#     rows = observed.shape[0]
#     cols = observed.shape[1]
#     df = (rows - 1) * (cols - 1)
#     p = 1 - chi2.cdf(chisq, df)
#     if p > rejection_factor:
#         print('p-value=%.5f, rejection_factor=%.5f:'%(p, rejection_factor),' fail to reject the null hypothesis')
#     else:
#         print('p-value=%.5f, rejection_factor=%.5f:'%(p, rejection_factor),' reject the null hypothesis')
    
#%% examples
#input data
import pandas as pd
fit=pd.read_csv('toy_data_observable_values.csv')
fit2=pd.read_csv('toy_data_observable_errors.csv')
df=pd.read_csv('toy_data_bin_0.csv')

ctk = df.ctk #toy data
fl=fit.fl_from_ctk[0] #fitted Fl from bin 0
ctk_fitted_error=fit2.fl_from_ctk[0] #error of fitted Fl from bin 0

ctk_fitted = 3/2 * fl * ctk**2 + 3/4 * (1-fl) * (1-ctk**2)
rejection_factor1d = 0.95
chisq(ctk_fitted,ctk,rejection_factor1d)

