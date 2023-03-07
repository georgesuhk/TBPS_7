# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:16:41 2023

@author: victo
"""

#multi-dimensional fitting - WIP

import numpy as np
import pandas as pd
import iminuit
import Efficiency_legendre_fit as acc
import real_mass_dist_fitting as mass


threshold = 0.9 # which threshold to use

data_path = f'thresholds/comb_threshold{threshold}/cleaned_td__peak0.9.csv'
acc_path = f'thresholds/comb_threshold{threshold}/cleaned_acc_peak0.9.csv'

#data_path = 'sig_monte_carlo.csv'
#acc_path = 'acceptance_mc.csv'

data = pd.read_csv(data_path)

if 'costhetal' in data.columns:
    data = data.rename(columns = {'costhetal': 'ctl'})
if 'costhetak' in data.columns:
    data = data.rename(columns = {'costhetak': 'ctk'})
    
    
mass.set_data(data_path)
acc.set_data(acc_path)

#values for initial guess
SM_vals = np.loadtxt('expected_observable_values.csv', delimiter = ',', skiprows = 1)[:,1:]

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

#functions for fitting

#not actually chebyshev but like they're so pointless just use any old quadratic
#we can convert the coefficients later

def full_func(cos_theta_l, cos_theta_k, phi, _bin, fsig, *params):
    '''
    The full probability distribution for all angles/observables

    Parameters
    ----------
    cos_theta_l : array of cos(theta_l) for all candidates
    cos_theta_k : array of cos(theta_k) for all candidates
    phi : array of phi for all candidates
    fsig : signal fraction
    *params : A_fb, F_l, S3, S4, S5, S7, S8, S9 (observables)
    more *params (background):
    c0_ctl, c1_ctl, c2_ctl: background parameterization in ctl
    c0_ctk, c1_ctk, c2_ctk: background parameterization in ctk
    c0_phi, c1_phi, c2_phi: background parameterization in phi
    

    Returns
    -------
    scalar_array : an array of values for the expression visible in the OneNote

    '''

    ctl = np.array(cos_theta_l)
    ctk = np.array(cos_theta_k)
    phi = np.array(phi)
    c2tl = 2 * ctl**2 - 1 # cos(2theta_l)
    stl2 = 1-(ctl**2) # sin^2(theta_l)
    stl = np.sqrt(stl2) # sin(theta_l)
    stk2 = 1-(ctk**2) # sin^2(theta_k)
    s2tl = 2 * ctl * stl # sin(2theta_l)
    s2tk = 2*ctk*np.sqrt(stk2) # sin(2theta_k)
    
    A_fb, F_l, S3, S4, S5, S7, S8, S9 = params #, c0_ctl, c1_ctl, c2_ctl, c0_ctk, c1_ctk, c2_ctk, c0_phi, c1_phi, c2_phi = params
    
    q2 = np.array([np.mean(bin_ranges[_bin]) for n in range(len(ctl))]).astype('float64')
    acceptance = acc.get_efficiency_kress(ctl, ctk, phi, q2, acc.coeffs, i_max=2, j_max=2, m_max=2, n_max=2)
    
    fbkg = 1-fsig
    
    # trying the better background
    
    bkg = fbkg/(8*np.pi)
    #bkg = fbkg * mass.bkg_dist(ctl, ctk, phi, *(mass.param_bins[_bin]))
    
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
    scalar_array *= fsig # normalize wrt background
    scalar_array += bkg
    scalar_array *= acceptance
    #scalar_array *= 2 # normalization after acceptance
    '''
    if np.any(scalar_array < 0):
        bad_ctl = np.array(ctl[scalar_array < 0]).reshape(1, np.sum(scalar_array < 0))
        bad_ctk = np.array(ctk[scalar_array < 0]).reshape(1, np.sum(scalar_array < 0))
        bad_phi = np.array(phi[scalar_array < 0]).reshape(1, np.sum(scalar_array < 0))
        bad_angles = np.append(np.append(bad_ctl, bad_ctk, axis=0), bad_phi, axis = 0)
        np.savetxt('full_func_err.csv', bad_angles, delimiter = ',')
        raise ValueError('y\'all need therapy')
    '''
    
    #scalar_array[scalar_array < 0] = np.exp(scalar_array[scalar_array<0]*500000)
    return scalar_array



def log_likelihood(_bin, fsig, *params):
    '''
    The negative log-likelihood of the probability distribution defined above with given parameters
    
    Parameters
    ----------
    *params: A_fb, F_l, S3, S4, S5, S7, S8, S9, + background parameterization in that order
    
    Returns: log likelihood of function with those parameters
    '''
    ctl = bins[_bin]['ctl']
    ctk = bins[_bin]['ctk']
    phi = bins[_bin]['phi']
    
    #limiting the value of ctk for testing purposes
    normalized_scalar_array = full_func(ctl, ctk, phi, _bin, fsig, *params)
    if np.any(normalized_scalar_array < 0):
        return 1e6 * np.sum(normalized_scalar_array < 0) # punish it by how many negative values come out - might be to flat since this will be discrete
    return - np.sum(np.log(normalized_scalar_array))


def minimize_logL(_bin, initial_guess):
    '''
    Minimizes the log-likelihood for the full pdf across all angles

    Parameters
    ----------
    _bin : the bin to fit to
    initial_guess : the initial guess of parameter values
        A_FB, F_L, S3, S4, S5, S7, S8, S9, + background parameterization in that order

    Returns
    -------
    m : minimized iminuit.Minuit object

    '''
    fsig = mass.find_fsig(_bin)
    min_func = lambda *args:log_likelihood(_bin, fsig, *args)
    min_func.errordef = iminuit.Minuit.LIKELIHOOD
    m = iminuit.Minuit(min_func, *initial_guess)
    m.limits = [(-1., 1.), (0., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.)]#, (0, 2), (-2, 2), (0, 2), (0, 2), (-2, 2), (0, 2), (0, 2), (-2, 2), (0, 2)]
    #m.fixed['x0'] = True
    m.migrad()
    m.hesse()
    
    return m

def get_observables(bin_number=len(bin_ranges)):
    vals = []
    errs = []
    for i in range(bin_number):
        initial_guess = list(SM_vals[i]) # + [1]*9
        m = minimize_logL(i, initial_guess)
        bin_vals = list(m.values)
        print(f'bin {i} minimum is valid: {m.fmin.is_valid}')
        bin_errs = list(m.errors)
        vals.append(bin_vals)
        errs.append(bin_errs)
    return np.array(vals), np.array(errs)

for i, _bin in enumerate(bins):
    initial_guess = list(SM_vals[i]) # + [1]*9
    positive = (full_func(_bin['ctl'], _bin['ctk'], _bin['phi'], i, mass.find_fsig(i), *initial_guess) > 0)
    bins[i] = _bin[positive]

#vals, errs = get_observables()




        
        