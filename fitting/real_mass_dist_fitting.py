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



 
bin_ranges = [[.1, .98],
              [1.1, 2.5],
              [2.5, 4.0],
              [4.0, 6.0],
              [6.0, 8.0],
              [15., 17.],
              [17., 19.],
              [11., 12.5],
              [1.0, 6.0],
              [15., 19.]]

#bin the data

def q2_binning_sm(data, bin_ranges):
    return [data[(data['q2'] >= bin_range[0]) & (data['q2'] <= bin_range[1])]['B0_M'] for bin_range in bin_ranges]

def q2_binning_ang(data, bin_ranges):
    return [data[(data['q2'] >= bin_range[0]) & (data['q2'] <= bin_range[1])][['ctl', 'ctk', 'phi']] for bin_range in bin_ranges]


def set_data(path):
    global mass_data
    global ang_data
    global bins
    global ang_bins
    global bin_params
    global bin_errs
    total_data = pd.read_csv(path)[['B0_M', 'q2', 'costhetal', 'costhetak', 'phi']]
    mass_data = total_data[['B0_M', 'q2']]
    ang_data = total_data[['costhetal', 'costhetak', 'phi', 'q2']]
    ang_data = ang_data.rename(columns = {'costhetal': 'ctl'})
    ang_data = ang_data.rename(columns = {'costhetak': 'ctk'})
    bins = q2_binning_sm(mass_data, bin_ranges)
    ang_bins = q2_binning_ang(ang_data, bin_ranges)
    bin_params = []
    bin_errs = []
    for _bin in range(len(bins)):
        p, e = find_background_params(_bin)
        bin_params.append(p)
        bin_errs.append(e)
        

total_data = pd.read_csv('thresholds/comb_threshold0.995/cleaned_td__peak0.9.csv')[['B0_M', 'q2', 'costhetal', 'costhetak', 'phi']]
mass_data = total_data[['B0_M', 'q2']]
ang_data = total_data[['costhetal', 'costhetak', 'phi', 'q2']]
ang_data = ang_data.rename(columns = {'costhetal': 'ctl'})
ang_data = ang_data.rename(columns = {'costhetak': 'ctk'})
bins = q2_binning_sm(mass_data, bin_ranges)
ang_bins = q2_binning_ang(ang_data, bin_ranges)


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
    A = (n/np.abs(alpha))**n * np.exp(-(alpha**2 / 2))
    B = (n/np.abs(alpha)) - np.abs(alpha)
    C = (n/np.abs(alpha)) * (1/(n-1)) * np.exp(-(np.abs(alpha))**2 / 2)
    D = np.sqrt(np.pi/2) * (1 + erf(np.abs(alpha))/np.sqrt(2))
    N = 1/(sigma * (C + D))
    
    #normalization
    a = np.min(x)
    b = np.max(x)
    
    power_law_term = ((N*A*-sigma)/(1-n)) * ((B+alpha)**(1-n) - (B-(a-mean)/sigma)**(1-n))
    gaussian_term = N*np.sqrt(2)*sigma*np.sqrt(np.pi)*0.5*(erf((b-mean)/(np.sqrt(2)*sigma)) - erf(-alpha/np.sqrt(2)))
    exponential_term = bkg_amp*tau*(1-np.exp((a-b)/tau))
    
    area = power_law_term + gaussian_term + exponential_term
    normalization = 1/area
    
    #distribution
    bkg = bkg_amp*np.exp(-(x-a)/tau)
    return (crystal_ball(x, mean, sigma, alpha, n) + bkg) * normalization

    
    
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
    if dist == bkg_dist or dist == bkg_dist_1st_order:
        ctl = np.array(ang_data['ctl'])
        ctk = np.array(ang_data['ctk'])
        phi = np.array(ang_data['phi'])
        scalar_array = dist(ctl, ctk, phi, *params)
    else:
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
        if dist == bkg_dist_1st_order:
            m.limits = [(0, 1), (-1, 1), (0, 1), (-1, 1), (0, 2), (-1/np.pi, 1/np.pi)]
        elif dist == gaussian_bkg:
            m.limits = [(5000, 5400), (0, 25), (.1, 10), (1.001, 5), (0, 10), (0, 1000)]
    else:
        m.limits = [(0, 1), (-2, 2), (-1, 2)] * 3
    m.migrad()
    m.hesse()
    return m



def mass_fit_gaussian(plotting = False):
    'Fits a gaussian distribution with background to data and returns fitted parameters'
    initial_guess = [5280, 15, 0.01, 70]
    m = minimize_logL(gaussian_bkg, initial_guess)
    #print(f'Minimum is valid: {m.fmin.is_valid}')
    if plotting:
        heights, edges = np.histogram(data, bins = 50)
        centers = 0.5*(edges[1] - edges[0]) + edges[:-1]
        ys = gaussian_bkg(centers, *m.values)
        check_params_plot(gaussian_bkg, m.values)
        #check_params_plot(gaussian_bkg, initial_guess)
    return np.array(m.values), np.array(m.errors)

def mass_fit_crystal(plotting = False):
    initial_guess = [5280, 16, 1, 2., 0.02, 70]
    m = minimize_logL(crystal_ball_bkg, initial_guess)
    #print(f'Minimum is valid: {m.fmin.is_valid}')
    if plotting:
        heights, edges = np.histogram(data, bins = 50)
        centers = 0.5*(edges[1] - edges[0]) + edges[:-1]
        ys = crystal_ball_bkg(centers, *m.values)
        check_params_plot(crystal_ball_bkg, m.values)
        #check_params_plot(crystal_ball_bkg, initial_guess)
    return np.array(m.values), np.array(m.errors)
    

def find_fsig_gaussian(_bin):
    global data
    if type(_bin) == type(None):
        data = mass_data['B0_M']
    else:
        data = bins[_bin]
    vals, errs = mass_fit_gaussian()
    xs = np.linspace(np.min(data), np.max(data), 2000)
    crystal_area = np.trapz(gaussian(xs, vals[0], vals[1]), xs)
    bkg_area = np.trapz(vals[2]*np.exp(-(xs-np.min(data))/vals[3]), xs)
    return crystal_area/(crystal_area + bkg_area)

def find_fsig_crystal(_bin):
    global data
    if type(_bin) == type(None):
        data = mass_data['B0_M']
    else:
        data = bins[_bin]
    vals, errs = mass_fit_crystal()
    xs = np.linspace(np.min(data), np.max(data), 2000)
    crystal_area = np.trapz(crystal_ball(xs, vals[0], vals[1], vals[2], vals[3]), xs)
    bkg_area = np.trapz(vals[4]*np.exp(-(xs-np.min(data))/vals[5]), xs)
    return crystal_area/(crystal_area + bkg_area)

def bkg_dist(ctl, ctk, phi, c0_ctl, c1_ctl, c2_ctl, c0_ctk, c1_ctk, c2_ctk, c0_phi, c1_phi, c2_phi):
    ctl_terms = c0_ctl + c1_ctl*ctl + c2_ctl*ctl**2
    ctk_terms = c0_ctk + c1_ctk*ctk + c2_ctk*ctk**2
    phi_terms = c0_phi + c1_phi*phi + c2_phi*phi**2
    ctl_norm = 2*c2_ctl/3 + 2*c0_ctl
    ctk_norm = 2*c2_ctk/3 + 2*c0_ctk
    phi_norm = (2/3)*c2_phi*np.pi**3 + 2*np.pi*c0_phi
    norm = ctl_norm*ctk_norm*phi_norm
    return ctl_terms*ctk_terms*phi_terms / norm

def bkg_dist_errs(ctl, ctk, phi, vals, errs):
    c0_ctl, c1_ctl, c2_ctl, c0_ctk, c1_ctk, c2_ctk, c0_phi, c1_phi, c2_phi = vals
    c0_ctl_err, c1_ctl_err, c2_ctl_err, c0_ctk_err, c1_ctk_err, c2_ctk_err, c0_phi_err, c1_phi_err, c2_phi_err = errs
    
    ctl_terms = c0_ctl + c1_ctl*ctl + c2_ctl*ctl**2
    ctk_terms = c0_ctk + c1_ctk*ctk + c2_ctk*ctk**2
    phi_terms = c0_phi + c1_phi*phi + c2_phi*phi**2
    
    ctl_errs = np.sqrt(c0_ctl_err**2 + (ctl*c1_ctl_err)**2 + (ctl**2*c2_ctl_err)**2)
    ctk_errs = np.sqrt(c0_ctk_err**2 + (ctk*c1_ctk_err)**2 + (ctk**2*c2_ctk_err)**2)
    phi_errs = np.sqrt(c0_phi_err**2 + (phi*c1_phi_err)**2 + (phi**2*c2_phi_err)**2)
    
    ctl_norm = 2*c2_ctl/3 + 2*c0_ctl
    ctk_norm = 2*c2_ctk/3 + 2*c0_ctk
    phi_norm = (2/3)*c2_phi*np.pi**3 + 2*np.pi*c0_phi
    norm = ctl_norm*ctk_norm*phi_norm
    
    variance = (ctl_errs*ctk_terms*phi_terms)**2 + (ctk_errs*ctl_terms*phi_terms)**2 + (phi_errs*ctk_terms*ctl_terms)**2
    
    return np.sqrt(variance)/norm
    
def bkg_dist_1st_order(ctl, ctk, phi, c0_ctl, c1_ctl, c0_ctk, c1_ctk, c0_phi, c1_phi):
    ctl_terms = c0_ctl + c1_ctl*ctl
    ctk_terms = c0_ctk + c1_ctk*ctk
    phi_terms = c0_phi + c1_phi*phi
    ctl_norm = 2*c0_ctl
    ctk_norm = 2*c0_ctk
    phi_norm = 2*np.pi*c0_phi
    norm = ctl_norm*ctk_norm*phi_norm
    return ctl_terms * ctk_terms * phi_terms / norm

    
def find_background_params(_bin):
    global data
    global ang_data
    if type(_bin) == type(None):
        data = mass_data['B0_M']
    else:
        data = bins[_bin]
        ang_data = ang_bins[_bin]
    vals, errs = mass_fit_gaussian()
    mean = vals[0]
    sd = vals[1]
    sideband = ang_bins[_bin][(data > (mean + 2*sd))]
    data = sideband
    initial_guess = [0.5, 0.2, 0.1]*3
    m = minimize_logL(bkg_dist, initial_guess)
    return np.array(m.values), np.array(m.errors)
    
   
def check_params_plot(dist, vals):
    'plots the distribution with the histogram to make sure the correct values have been found'
    heights, edges, patches = plt.hist(data, bins = 50, color = 'xkcd:light blue')
    centers = 0.5*(edges[1] - edges[0]) + edges[:-1]
    xs = np.linspace(centers.min(), centers.max(), 1000)
    ys = dist(xs, *vals)
    scale = np.trapz(heights, centers) / np.trapz(ys, xs)
    plt.plot(xs, ys*scale)
    
find_fsig = find_fsig_gaussian

