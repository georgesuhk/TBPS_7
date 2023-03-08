#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:27:39 2023

@author: moriahsanusi
"""
#Uses Method 2 to calculate the optimised observables by transforming the already fitted angular 
#observables into the optimised observables 


import numpy as np
import pandas as pd
import iminuit
import Efficiency_legendre_fit as acc
import fitting_visualization as fv

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
    ctl = np.array(cos_theta_l)
    ctk = np.array(cos_theta_k)
    phi = np.array(phi)
    c2tl = 2 * ctl**2 - 1 # cos(2theta_l)
    stl2 = 1-(ctl**2) # sin^2(theta_l)
    stl = np.sqrt(stl2) # sin(theta_l)
    stk2 = 1-(ctk**2) # sin^2(theta_k)
    s2tl = 2*ctl*stl # sin(2theta_l)
    s2tk = 2*ctk*np.sqrt(stk2) # sin(2theta_k)
    
    A_fb, F_l, S3, S4, S5, S7, S8, S9 = params
    q2 = np.array([np.mean(bin_ranges[_bin]) for n in range(len(ctl))]).astype('float64')
    acceptance = acc.get_efficiency_kress(ctl, ctk, phi, q2, acc.coeffs)
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
    
    Parametersd
    ----------
    *params: A_fb, F_l, S3, S4, S5, S7, S8, S9 in that order
    
    Returns: log likelihood of function with those parameters
    '''
    ctl = bins[_bin]['costhetal']
    ctk = bins[_bin]['costhetak']
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
    m.limits = [(-1., 1.), (0., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.)]
    #m.fixed['x0'] = True
    m.migrad()
    m.hesse()
    
    return m

def get_observables(bin_number=len(bin_ranges)):
    vals = []
    errs = []
    for i in range(bin_number):
        m = minimize_logL(i, [0., 0.2, 0., 0., 0., 0., 0., 0.])
        bin_vals = list(m.values)
        print(f'bin {i} minimum is valid: {m.fmin.is_valid}')
        bin_errs = list(m.errors)
        vals.append(bin_vals)
        errs.append(bin_errs)
    return np.array(vals), np.array(errs)


#vals, errs = get_observables()

#function to output the optimised observables- F_L, P_i's 

def get_opt_observables(values, errors):
    """
    This function converts the fitted angular observables found using the get_observables() function and 
    transforms them into the optimised observables

    Parameters
    ----------
    values : arr
        Array of the fitted angular observables found from the get_observables() function for the
        different bin ranges
    errors : arr
        Array of the errors of the fitted angular observables found from the get_observables() 
        function for the different bin ranges

    Returns
    -------
    Array of the fitted optimised values and an array of the errors of the fitted optimised values 

    """
    
   # p_vals = vals #creating a copy of vals 
   #extracting the angular observables

    afb = []
    fl = []
    s3 = []
    s4 = []
    s5 = []
    s7 = []   
    s8 = [] 
    s9 = []   
    
    for bins in values:
        afb.append(bins[0])
        fl.append(bins[1])
        s3.append(bins[2])
        s4.append(bins[3])
        s5.append(bins[4])
        s7.append(bins[5])
        s8.append(bins[6])
        s9.append(bins[7])
        
    afb = np.array(afb)
    fl = np.array(fl)
    s3 = np.array(s3)
    s4 = np.array(s4)
    s5 = np.array(s5)
    s7 = np.array(s7)
    s8 = np.array(s8)
    s9 = np.array(s9)
      #transforming the angular observables to optimised observables
    
    p1= (2*s3)/(1-fl)
    p2 = (2/3)*afb*(1/(1-fl))
    p3 = -(1/(1-fl))*s9
    p4p = (1/(np.sqrt(fl*(1-fl))))*s4
    p5p = (1/(np.sqrt(fl*(1-fl))))*s5
    p6p = (1/(np.sqrt(fl*(1-fl))))*s7
    p8p = 1/(np.sqrt(fl*(1-fl)))*s8

    #adding them to a new pandas dataframe 
    
# calculating the errors
    afb_err = []
    fl_err = []
    s3_err = []
    s4_err = []
    s5_err = []
    s7_err = []   
    s8_err = [] 
    s9_err = []      

    for bins in errors:
        afb_err.append(bins[0])
        fl_err.append(bins[1])
        s3_err.append(bins[2])
        s4_err.append(bins[3])
        s5_err.append(bins[4])
        s7_err.append(bins[5])
        s8_err.append(bins[6])
        s9_err.append(bins[7])
       
    afb_err = np.array(afb_err)
    fl_err = np.array(fl_err)
    s3_err = np.array(s3_err)
    s4_err = np.array(s4_err)
    s5_err = np.array(s5_err)
    s7_err = np.array(s7_err)
    s8_err = np.array(s8_err)
    s9_err = np.array(s9_err)
    

    p1_errs = 2*p1*np.sqrt((s3_err/s3)**2 +(fl_err/fl)**2)
    p2_errs = p2*(2/3)*np.sqrt((afb_err/afb)**2 +(fl_err/fl)**2)
    p3_errs = p3*(-1*np.sqrt((s9_err/s9)**2 +(fl_err/fl)**2))
    fl_sq_err = (fl**2)*np.sqrt((fl_err/fl)**2+(fl_err/fl)**2)
    fl_sqrt_err = 0.5*((fl*(1-fl))**(-0.5))*fl_sq_err
    p4p_errs = p4p*np.sqrt(((s4_err/s4)**2) + (fl_sqrt_err/np.sqrt(fl*(1-fl))))
    p5p_errs = p5p*np.sqrt(((s5_err/s4)**2) + (fl_sqrt_err/np.sqrt(fl*(1-fl))))    
    p8p_errs = p8p*np.sqrt(((s8_err/s4)**2) + (fl_sqrt_err/np.sqrt(fl*(1-fl))))    
    p6p_errs = p6p*np.sqrt(((s7_err/s4)**2) + (fl_sqrt_err/np.sqrt(fl*(1-fl))))    
    
    
    opt_d = {'fl': fl, 'p1': p1, 'p2':p2, 'p3':p3, 'p4p': p4p, 'p5p': p5p, 'p6p': p6p, 'p8p': p8p}
    opt_df = pd.DataFrame(opt_d, index = [0,1,2,3,4,5,6,7,8,9])
    
    opt_err_d = {'fl_err': fl_err, 'p1_err': p1_errs, 'p2_err':p2_errs, 'p3_err':p3_errs, 'p4p_err': p4p_errs, 'p5p_err': p5p_errs, 'p6p_err': p6p_errs, 'p8p_err': p8p_errs}
    opt_err_df = pd.DataFrame(opt_err_d, index = [0,1,2,3,4,5,6,7,8,9])
    return np.array(opt_df), np.array(opt_err_df)


s_vals, s_errs = get_observables()
opt_vals_m2, opt_errs_m2 = get_opt_observables(s_vals, s_errs)

opt_labels = ['F_L', 'P1', 'P2', 'P3', 'P4p', 'P5p', 'P6p', 'P8p']
opt_errs_labels = ['F_L_err', 'P1_err', 'P2_err', 'P3_err', 'P4p_err', 'P5p_err', 'P6p_err', 'P8p_err']
# create a pandas dataframe
opt_vals_df_m2 = pd.DataFrame(opt_vals_m2, columns=opt_labels)
opt_errs_df_m2 = pd.DataFrame(opt_errs_m2, columns = opt_errs_labels)
# save the dataframe to a CSV file
opt_vals_df_m2.to_csv('opt_vals_M2.csv', index=True)
opt_errs_df_m2.to_csv('opt_errs_M2.csv', index=True)

fv.plot_p_observables(opt_vals_m2, opt_errs_m2, bin_ranges)

