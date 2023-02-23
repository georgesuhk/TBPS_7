# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:57:21 2023

@author: victo
"""

#real mass dist fitting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iminuit
from scipy.special import erf

data = pd.read_csv('thresholds/comb_threshold0.75/cleaned_td__peak0.9.csv')['B0_M']

#mass distribution functions

def gaussian(x, mean, sigma):
    'gaussian innit'
    return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*((x-mean)/sigma)**2)

def gaussian_bkg(x, mean, sigma, bkg_amp, tau):
    'normalized gaussian with background exponential'
    #normalization
    a = np.min(x)
    b = np.max(x)
    # see OneNote for justification of normalization
    z_1 = (a-mean)/(np.sqrt(2)*sigma)
    z_2 = (b-mean)/(np.sqrt(2)*sigma)
    N = 0.5 * (erf(z_2) - erf(z_1)) + bkg_amp*tau*(1 - np.exp(-(b-a)/tau))
    #distribution
    bkg = bkg_amp*np.exp(-(x-np.min(data))/tau)
    return (gaussian(x, mean, sigma) + bkg)/N
    
def crystal_ball(x, mean, sigma, alpha, n):
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
    
def crystal_ball_bkg(x, mean, sigma, alpha, n, bkg_amp, tau):
    '''
    The Crystal Ball function with a background term
    
    WARNING: the minimum of x must be in the power law region and the maximum in the Gaussian region
    
    If you do not do this, it will break the normalization and the optimizer will be useless
    '''
    #normalization
    a = np.min(x)
    b = np.max(x)
    
    A = (n/np.abs(alpha))**n * np.exp(-(alpha**2 / 2))
    B = (n/np.abs(alpha)) - np.abs(alpha)
    C = (n/np.abs(alpha)) * (1/(n-1)) * np.exp(-(np.abs(alpha))**2 / 2)
    D = np.sqrt(np.pi/2) * (1 + erf(np.abs(alpha))/np.sqrt(2))
    N = 1/(sigma * (C + D))
    
    power_law_term = (N*A)/(-n+1) * ((B+alpha)**(-n+1) - (B-((a-mean)/sigma))**(-n+1))
    gaussian_term = N*np.sqrt(2)*sigma*np.sqrt(np.pi)*0.5*(erf((b-mean)/(np.sqrt(2)*sigma)) - erf(-alpha/2))
    exponential_term = bkg_amp*tau*(1-np.exp((a-b)/tau))
    
    area = power_law_term + gaussian_term + exponential_term
    normalization = 1/area
    
    #distribution
    bkg = bkg_amp*np.exp(-(x-np.min(data))/tau)
    return (crystal_ball(x, mean, alpha, n, sigma) + bkg) * normalization
    
    
def log_likelihood(dist, *params):
    '''
    The log-likelihood function for a mass distribution with given parameters based on the data in the given bin
    
    Parameters:
        dist: the distribution to fit to the data
        _bin: the bin to which the distribution is fitted
        *params: the parameters for the distribution
    
    Returns:
        scalar negative log-likelihood value
    '''
    scalar_array = dist(np.array(data), *params) # use np.array(mass_data) because of jankiness between pandas and numpy
    
    return -np.sum(np.log(scalar_array))

def minimize_logL(dist, initial_guess):
    '''
    Minimizes the log-likelihood for a given distribution fitted to the mass data of a given bin'
    
    Parameters:
        dist: the distribution for which to minimize log-likelihood
        _bin: the data to which the distribution is fitted
        *params: the initial guess for the distribution
    '''
    min_func = lambda *args:log_likelihood(dist, *args)
    min_func.errordef = iminuit.Minuit.LIKELIHOOD
    m = iminuit.Minuit(min_func, *initial_guess)
    if len(initial_guess) == 4:
        m.limits = [(5000, 5400), (0, 25), (0, 10), (0, 1000)]
    elif len(initial_guess) == 6:
        m.limits = [(5000, 5400), (0, 25), (.1, 10), (1.001, 5), (0, 10), (0, 1000)]
    m.migrad()
    m.hesse()
    return m



def mass_fit_gaussian(plotting = False):
    'Fits a gaussian distribution with background to data and returns fitted parameters'
    initial_guess = [5280, 15, 0.01, 70]
    m = minimize_logL(gaussian_bkg, initial_guess)
    print(f'Minimum is valid: {m.fmin.is_valid}')
    if plotting:
        heights, edges = np.histogram(data, bins = 50)
        centers = 0.5*(edges[1] - edges[0]) + edges[:-1]
        ys = gaussian_bkg(centers, *m.values)
        print(np.trapz(ys, centers))
        check_params_plot(gaussian_bkg, m.values)
        check_params_plot(gaussian_bkg, initial_guess)
    return np.array(m.values), np.array(m.errors)

def mass_fit_crystal(plotting = False):
    initial_guess = [5280, 16, 2, 1.5, 0.02, 70]
    m = minimize_logL(crystal_ball_bkg, initial_guess)
    print(f'Minimum is valid: {m.fmin.is_valid}')
    if plotting:
        heights, edges = np.histogram(data, bins = 50)
        centers = 0.5*(edges[1] - edges[0]) + edges[:-1]
        ys = crystal_ball_bkg(centers, *m.values)
        print(np.trapz(ys, centers))
        check_params_plot(crystal_ball_bkg, m.values)
        check_params_plot(crystal_ball_bkg, initial_guess)
    return np.array(m.values), np.array(m.errors)
    
    
def check_params_plot(dist, vals):
    'plots the distribution with the histogram to make sure the correct values have been found'
    heights, edges, patches = plt.hist(data, bins = 50, color = 'xkcd:light blue')
    centers = 0.5*(edges[1] - edges[0]) + edges[:-1]
    xs = np.linspace(centers.min(), centers.max(), 1000)
    ys = dist(xs, *vals)
    scale = np.trapz(heights, centers) / np.trapz(ys, xs)
    plt.plot(xs, ys*scale)