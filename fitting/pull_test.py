#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:18:27 2023

@author: fei
"""

import numpy as np

def pull_test(expected,observed,distance):
    d = (observed-expected).flatten()
    count=0
    for i in range (0,len(d)):
        if np.absolute(d[i])<distance:
            count+=1
    return print('Percentage rate of passing pull test=',count/len(d))
    
#%% examples
#input data
expected1d = np.array([50, 56, 50, 50, 50])
observed1d = np.array([50, 50, 50, 50, 51])
distance = 5
pull_test(expected1d,observed1d,distance)

#input data
expected2d = np.array([[50, 50, 50, 50, 50],
                     [50, 50, 50, 50, 50],
                     [50, 50, 50, 50, 50]])
observed2d = np.array([[50, 60, 40, 47, 53],
                     [50, 66, 40, 47, 53],
                     [40, 60, 40, 47, 53]])
distance = 5
pull_test(expected2d,observed2d,distance)