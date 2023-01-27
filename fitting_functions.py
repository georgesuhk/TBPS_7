# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:29:45 2023

@author: victor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#remember to have the correct path for this part
files = [f'toy_data_bin_{i}.csv' for i in range(7)]
bins = [pd.read_csv(file) for file in files]

bins[0].head()

def projection_ctk(cos_theta_k, fl):
    '''
    Returns the projection of the pdf onto cos(theta_k) 
    '''
    acceptance = 0.5 # placeholder acceptance!!!
    scalar_array = (3/4) * ((1-fl)*(1-cos_theta_k**2) + 2 * fl * cos_theta_k**2) * acceptance
    normalized_scalar_array = scalar_array / np.sum(scalar_array) # normalizes the pdf
    return normalized_scalar_array


def projection_ctl(cos_theta_l, fl, afb):
    """
    Returns the projection of the pdf onto cos(theta_l)
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    acceptance = 0.5  # acceptance "function"
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance
    normalized_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    return normalized_scalar_array

def projection_ctk(phi, s3, s9, fl):
    '''
    Returns the projection of the pdf onto cos(theta_k) 
    '''
    acceptance = 0.5 # placeholder acceptance!!!
    scalar_array = (1/np.pi) * (1 + s3*np.cos(phi)**2 + s9*np.sin(phi)**2) * acceptance
    normalized_scalar_array = scalar_array / np.sum(scalar_array) # normalizes the pdf
    return normalized_scalar_array

def log_likelihood(func, fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    _bin = bins[int(_bin)]
    ctl = _bin['ctl']
    normalized_scalar_array = func(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalized_scalar_array))

