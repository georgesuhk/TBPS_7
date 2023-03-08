#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:37:06 2023

@author: moriahsanusi
"""
#uses Method 1 to find the optimised observables by directly fitting the optimised observable parameters to
#the angular distributions equation in the OneNote

import numpy as np
import pandas as pd
import iminuit
import Efficiency_legendre_fit as acc
import fitting_visualization as fv
import particle_antiparticle_acceptance as paa

#%%
# Method 1 where the equation for the optimised observables was used, the optimised observables
#were fitted, the minimum log liklihood was found and the minimisation took place
data = pd.read_csv('cleaned_total_dataset.csv')

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
    return [data[(data['q2'] >= bin_range[0]) & (data['q2'] <= bin_range[1])] for bin_range in bin_ranges]

bins = q2_binning_sm(data, bin_ranges)



def opt_full_func(cos_theta_l, cos_theta_k, phi, _bin, *params):
    """
    

    Parameters
    ----------
    cos_theta_l : array of cos(theta_l) for all candidates
    cos_theta_k : array of cos(theta_k) for all candidates
    phi : array of phi variables for all candidates
        
    *params : TYPE
        F_l, P1, P2, P3, P4P, P5P, P6P, P8P

    Returns
    -------
    scalar_array : TYPE
        scalar_array : an array of values for the expression visible in the OneNote with the p observables

    """
    ctl = np.array(cos_theta_l)
    ctk = np.array(cos_theta_k)
    phi = np.array(phi)
    c2tl = 2 * ctl**2 - 1 # cos(2theta_l)
    stl2 = 1-(ctl**2) # sin^2(theta_l)
    stl = np.sqrt(stl2) # sin(theta_l)
    stk2 = 1-(ctk**2) # sin^2(theta_k)
    s2tl = 2*ctl*stl # sin(2theta_l)
    s2tk = 2*ctk*np.sqrt(stk2) # sin(2theta_k)
    
    F_l, P1, P2, P3, P4P, P5P, P6P, P8P = params
    
    #writing the angular observables in terms of the optimised observables 
    A_fb = (3/2)*(1. - F_l)* P2
    S3 = 0.5*(1. - F_l)*P1
    S4 = (np.sqrt(F_l*(1-F_l)))*P4P
    S5 = (np.sqrt(F_l*(1-F_l)))*P5P
    S7 = (np.sqrt(F_l*(1-F_l)))*P6P
    S8 = (np.sqrt(F_l*(1-F_l)))*P8P
    S9 = -(1. - F_l)*P3
    
    q2 = np.array([np.mean(bin_ranges[_bin]) for n in range(len(ctl))]).astype('float64')
    acceptance = paa.acceptance_all(ctl, ctk, phi, q2)
    
    A_fb_term = A_fb * (4/3) * stk2 * ctl
    Fl_term1 = (3/4) * (1. - F_l) * stk2
    Fl_term2 = F_l * ctk**2
    Fl_term3 = (1/4) * (1. - F_l) * stk2 * c2tl
    Fl_term4 =  - F_l * (ctk**2) * c2tl
    S3_term = S3 * stk2 * stl2 * np.cos(2*phi)
    S4_term = S4 * s2tk * s2tl * np.cos(phi)
    S5_term = S5 * s2tk * stl * np.cos(phi)
    S7_term = S7 * s2tk * stl * np.sin(phi)
    S8_term = S8 * s2tk * s2tl * np.sin(phi)
    S9_term = S9 * stk2 * stl2 * np.sin(2*phi)
    
    scalar_array = (9/(32*np.pi)) * (A_fb_term + Fl_term1 + Fl_term2 + Fl_term3 + Fl_term4 + S3_term + S4_term + S5_term + S7_term + S8_term + S9_term)
    scalar_array *= acceptance   
    
    return scalar_array


def opt_log_likelihood(_bin, *params):
    '''
    The negative log-likelihood of the probability distribution of the optimised observables defined above with given parameters
    
    Parameters
    ----------
    *params: F_l, P1, P2, P3, P4P, P5P, P6P, P8P in that order
    
    Returns: log likelihood of function with those parameters
    '''
    ctl = bins[_bin]['costhetal']
    ctk = bins[_bin]['costhetak']
    phi = bins[_bin]['phi']
    
    #limiting the value of ctk for testing purposes
    
    normalized_scalar_array = opt_full_func(ctl, ctk, phi, _bin, *params)
    if np.any(normalized_scalar_array < 0):
        return 1e6 * np.sum(normalized_scalar_array < 0) # punish it by how many negative values come out - might be to flat since this will be discrete
    return - np.sum(np.log(normalized_scalar_array))


def opt_minimize_logL(_bin, initial_guess):
    '''
    Minimizes the log-likelihood for the full optimised observables pdf across all angles

    Parameters
    ----------
    _bin : the bin to fit to
    initial_guess : the initial guess of parameter values
    F_l, P1, P2, P3, P4P, P5P, P6P, P8P in that order

    Returns
    -------
    m : minimized iminuit.Minuit object

    '''
    min_func = lambda *args:opt_log_likelihood(_bin, *args)
    min_func.errordef = iminuit.Minuit.LIKELIHOOD
    m = iminuit.Minuit(min_func, *initial_guess)
    m.limits = [(0., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.)]
    #m.fixed['x0'] = True
    m.migrad()
    m.hesse()
    
    return m
    
def opt_get_observables(bin_number=len(bin_ranges)):
    """
    Function to obtain the fitted parameter values for the optimised observables 

    Parameters
    ----------
    bin_number : TYPE, optional
        DESCRIPTION. The default is len(bin_ranges).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    vals = []
    errs = []
    for i in range(bin_number):
        m = opt_minimize_logL(i, [0.2, 0., 0., 0., 0., 0.,0.,0.])
        bin_vals = list(m.values)
        print(f'bin {i} minimum is valid: {m.fmin.is_valid}')
        bin_errs = list(m.errors)
        vals.append(bin_vals)
        errs.append(bin_errs)
    return np.array(vals), np.array(errs)


#opt_vals, opt_errs = opt_get_observables()

#getting the 
opt_vals_m1, opt_errs_m1 = opt_get_observables()

opt_labels = ['F_L', 'P1', 'P2', 'P3', 'P4p', 'P5p', 'P6p', 'P8p']
opt_errs_labels = ['F_L_err', 'P1_err', 'P2_err', 'P3_err', 'P4p_err', 'P5p_err', 'P6p_err', 'P8p_err']
# create a pandas dataframe
opt_vals_df_m1 = pd.DataFrame(opt_vals_m1, columns=opt_labels)
opt_errs_df_m1 = pd.DataFrame(opt_errs_m1, columns = opt_errs_labels)
# save the dataframe to a CSV file
opt_vals_df_m1.to_csv('opt_vals_M12.csv', index=True)
opt_errs_df_m1.to_csv('opt_errs_M12.csv', index=True)

fv.plot_p_observables(opt_vals_m1, opt_errs_m1, bin_ranges)


