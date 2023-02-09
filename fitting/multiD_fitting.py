# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 12:16:41 2023

@author: victo
"""

#multi-dimensional fitting - WIP

import numpy as np
import pandas as pd
import iminuit
import efficiency_fit as acc

data_path = 'C:/Users/victo/ic-teach-kstmumu-public/kstarmumu_toy_data/'
#remember to have the correct path for this part
#currently accessing toy data
files = [f'{data_path}toy_data_bin_{i}.csv' for i in range(7)]
bins = [pd.read_csv(file) for file in files]



def full_func(cos_theta_l, cos_theta_k, phi, _bin, *params):
    '''
    The full probability distribution for all angles/observables

    Parameters
    ----------
    cos_theta_l : array of cos(theta_l) for all candidates
    cos_theta_k : array of cos(theta_k) for all candidates
    phi : array of phi for all candidates
    *params : A_fb, F_l, S3, S4, S5, S7, S8, S9 in that order

    Returns
    -------
    scalar_array : an array of values for the expression visible in the OneNote

    '''
    ctl = cos_theta_l
    ctk = cos_theta_k
    c2tl = 2 * ctl**2 - 1 # cos(2theta_l)
    stl2 = 1-(ctl**2) # sin^2(theta_l)
    stl = np.sqrt(stl2) # sin(theta_l)
    stk2 = 1-(ctk**2) # sin^2(theta_k)
    s2tl = 2*ctl*stl # sin(2theta_l)
    s2tk = 2*ctk*np.sqrt(stk2) # sin(2theta_k)
    
    A_fb, F_l, S3, S4, S5, S7, S8, S9 = params
    acceptance = acc.n6_polynomial(ctk, *acc.popt_costhetak_ls[_bin])*acc.n6_polynomial_even(ctl, *acc.popt_costhetal_ls[_bin])*acc.n6_polynomial_even(phi, *acc.popt_phi_ls[_bin])
    A_fb_term = A_fb * (4/3) * stk2 * ctl
    Fl_term1 = (3/4) * (1 - F_l) * stk2
    Fl_term2 = F_l * ctk**2
    Fl_term3 = (1/4) * (1 - F_l) * stk2 * c2tl
    Fl_term4 =  - F_l * (ctk**2) * c2tl
    S3_term = S3 * stk2 * stl2 * np.cos(2*phi)
    S4_term = S4 * s2tk * s2tl * np.cos(phi)
    S5_term = S5 * s2tk * stl * np.cos(phi)
    S7_term = S7 * s2tk * stl * np.sin(phi)
    S8_term = S8 * s2tk * s2tl * np.sin(phi)
    S9_term = S9 * stk2 * stl2 * np.sin(2*phi)
    
    scalar_array = (9/(32*np.pi)) * (A_fb_term + Fl_term1 + Fl_term2 + Fl_term3 + Fl_term4 + S3_term + S4_term + S5_term + S7_term + S8_term + S9_term)
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



def log_likelihood(_bin, *params):
    '''
    The negative log-likelihood of the probability distribution defined above with given parameters
    
    Parameters
    ----------
    *params: A_fb, F_l, S3, S4, S5, S7, S8, S9 in that order
    
    Returns: log likelihood of function with those parameters
    '''
    ctl = bins[_bin]['ctl']
    ctk = bins[_bin]['ctk']
    phi = bins[_bin]['phi']
    
    #limiting the value of ctk for testing purposes
    
    normalized_scalar_array = full_func(ctl, ctk, phi, _bin, *params)
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
        A_FB, F_L, S3, S4, S5, S7, S8, S9 in that order

    Returns
    -------
    m : minimized iminuit.Minuit object

    '''
    min_func = lambda *args:log_likelihood(_bin, *args)
    min_func.errordef = iminuit.Minuit.LIKELIHOOD
    m = iminuit.Minuit(min_func, *initial_guess)
    m.limits = [(-1, 1), (0, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    #m.fixed['x0'] = True
    m.migrad()
    m.hesse()
    
    return m

def get_observables(bin_number=7):
    vals = []
    errs = []
    for i in range(bin_number):
        m = minimize_logL(i, [0, 0.2, 0, 0, 0, 0, 0, 0])
        bin_vals = list(m.values)
        print(f'bin {i} minimum is valid: {m.fmin.is_valid}')
        bin_errs = list(m.errors)
        vals.append(bin_vals)
        errs.append(bin_errs)
    return np.array(vals), np.array(errs)
        
