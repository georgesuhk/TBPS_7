# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 21:24:56 2023

@author: victo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iminuit
from scipy.special import erf


data_path = 'C:/Users/victo/ic-teach-kstmumu-public/kstarmumu_toy_data/'
#remember to have the correct path for this part
#currently accessing toy data
files = [f'{data_path}toy_data_bin_{i}.csv' for i in range(7)]
bins = [pd.read_csv(file) for file in files]

#mass distribution functions
def crystal_ball(x, mean, alpha, n, sigma):
    '''
    The 'Crystal Ball' function for the mass distribution
    
    Parameters:
        x : domain of the function
        mean : mean of the Gaussian
        alpha : cutoff between power law and Gaussian
        n : power law exponent
        sd : standard deviation of the Gaussian
    Returns: value at x
    '''
    
    A = (n/np.abs(alpha))**n * np.exp(-(alpha**2 / 2))
    B = (n/np.abs(alpha)) - np.abs(alpha)
    C = (n/np.abs(alpha)) * (1/(n-1)) * np.exp(-(np.abs(alpha))**2 / 2)
    D = np.sqrt(np.pi/2) * (1 + erf(np.abs(alpha))/np.sqrt(2))
    N = 1/(sigma * (C + D))
    
    gaussian_indices = np.where(((x-mean)/sigma > -alpha))
    power_law_indices = np.where(((x-mean)/sigma <= -alpha))
    results = np.zeros(x.shape)
    results[gaussian_indices] = np.exp(-((x[gaussian_indices]-mean)**2)/(2*sigma**2))
    results[power_law_indices] = A * (B - (x[power_law_indices]-mean)/sigma)**(-n)
    
    return N * results
    
    
def mB_dist(x, mean, alpha1, alpha2, n1, n2, sigma1, sigma2):
    'Suspected distribution of the B0 masses - two superposed crystal_ball functions w. same parameters each'
    return (crystal_ball(x, mean, alpha1, n1, sigma1) + crystal_ball(x, mean, alpha2, n2, sigma2))/2   



def log_likelihood(dist, _bin, *params):
    '''
    The log-likelihood function for a mass distribution with given parameters based on the data in the given bin
    
    Parameters:
        dist: the distribution to fit to the data
        _bin: the bin to which the distribution is fitted
        *params: the parameters for the distribution
    
    Returns:
        scalar negative log-likelihood value
    '''
    mass_data = bins[_bin]['mB']
    scalar_array = dist(np.array(mass_data), *params) # use np.array(mass_data) because of jankiness between pandas and numpy
    
    return -np.sum(np.log(scalar_array))

def minimize_logL(dist, _bin, initial_guess):
    '''
    Minimizes the log-likelihood for a given distribution fitted to the mass data of a given bin'
    
    Parameters:
        dist: the distribution for which to minimize log-likelihood
        _bin: the data to which the distribution is fitted
        *params: the initial guess for the distribution
    '''
    min_func = lambda *args:log_likelihood(dist, _bin, *args)
    min_func.errordef = iminuit.Minuit.LIKELIHOOD
    m = iminuit.Minuit(min_func, *initial_guess)
    m.migrad()
    m.hesse()
    return m


def fit_crystal_ball(_bin, plotting=False):
    'Fits a crystal ball distribution to the specified bin and returns an array of fitted parameters'
    initial_guess = [200, 1, 2., 10]
    m = minimize_logL(crystal_ball, _bin, initial_guess)
    if plotting:
        heights, edges, patches = plt.hist(bins[_bin]['mB'])
        centers = 0.5*(edges[1] - edges[0]) + edges[:-1]
        xs = np.linspace(centers.min(), centers.max(), 1000)
        ys = crystal_ball(xs, *m.values)
        scale = np.trapz(heights, centers) / np.trapz(ys, xs)
        plt.plot(xs, ys*scale)
    #print(f'Minimum is valid: {m.fmin.is_valid}')
    return m.values, m.errors

def fit_toy_data_masses(plotting = False):
    'Fit crystal ball distributions to each bin of the toy data and returns an array of fitted parameters'
    vals = []
    errs = []
    if plotting:
        fig, ax = plt.subplots(1, len(bins))
    for i, _bin in enumerate(bins):
        if plotting:
            plt.subplot(1, len(bins), i+1)
            plt.title('Bin %.i' %i)
            plt.xlabel(r'$m_b$')
            bin_vals, bin_errs = fit_crystal_ball(i, plotting=True)
        else:
            bin_vals, bin_errs = fit_crystal_ball(i, plotting=False)
        vals.append(bin_vals)
        errs.append(bin_errs)
    return np.array(vals), np.array(errs)
       
    