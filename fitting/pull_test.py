#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:18:27 2023

@author: fei
"""
import numpy as np
import matplotlib.pyplot as plt
def pull_test(expected,observed,observed_sd,nbins):
    pull = ((observed-expected)/observed_sd).flatten()
    return plt.hist(pull, bins=nbins)
    
#%% examples
#input data
expected1d = np.array([50, 56, 50, 50, 50])
observed1d = np.array([50, 50, 50, 50, 51])
observed1d_sd = np.array([1, 1, 2, 5, 2])
nbins = 3
pull_test(expected1d,observed1d,observed1d_sd,nbins)
plt.show()
#input data
expected2d = np.array([[50, 50, 50, 50, 50],
                     [50, 50, 50, 50, 50],
                     [50, 50, 50, 50, 50]])
observed2d = np.array([[50, 60, 40, 47, 53],
                     [50, 66, 40, 47, 53],
                     [40, 60, 40, 47, 53]])
observed2d_sd = np.array([[5, 6, 3, 4, 3],
                     [0, 6, 4, 7, 3],
                     [4, 6, 4, 4, 3]])
nbins = 3
pull_test(expected2d,observed2d,observed2d_sd,nbins)
plt.show()

