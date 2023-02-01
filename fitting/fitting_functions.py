# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:29:45 2023

@author: victor
"""

import numpy as np
import pandas as pd
import iminuit
import copy

data_path = 'C:/Users/victo/ic-teach-kstmumu-public/kstarmumu_toy_data/'
#remember to have the correct path for this part
#currently accessing toy data
files = [f'{data_path}toy_data_bin_{i}.csv' for i in range(7)]
bins = [pd.read_csv(file) for file in files]



#Probability distributions to fit to the data
  
def pdf_ctk(cos_theta_k, afb=0, fl=0, s3=0, s9=0):
    '''
    The projection of the pdf onto cos(theta_k) 
    
    Only parameters relevant to the function are listed in the docstring;
    this is for a general log-likelihood function
    
        cos_theta_k: cos(theta_l)
        fl: f_l observable
    
    Returns: d2gamma_P/dq^2dcos(theta_k)
    '''
    acceptance = 0.5 # placeholder acceptance!!!
    scalar_array = (3/4) * ((1-fl)*(1-cos_theta_k**2) + 2 * fl * cos_theta_k**2) * acceptance
    normalized_scalar_array = scalar_array *2 # normalizes the pdf
    return np.abs(normalized_scalar_array) # returning absolute value to prevent negative values in the log-likelihood sum


def pdf_ctl(cos_theta_l, afb=0, fl=0, s3=0, s9=0):
    '''
    The projection of the pdf onto cos(theta_l)
    
    Only parameters relevant to the function are listed in the docstring;
    this is for a general log-likelihood function
    Parameters:
        cos_theta_l: cos(theta_l)
        fl: f_l observable
        afb: a_fb observable
    
    Returns: d2gamma_P/dq^2dcos(theta_l)
    '''
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    acceptance = 0.5  # acceptance "function"
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance
    normalized_scalar_array = scalar_array * 2 # normalising scalar array to account for the non-unity acceptance function
    return np.abs(normalized_scalar_array) # returning absolute value to prevent negative values in the log-likelihood sum

def pdf_phi(phi, afb=0, fl=0, s3=0, s9=0):
    '''
    The projection of the pdf onto Phi
    
    Only parameters relevant to the function are listed in the docstring;
    this is for a general log-likelihood function
    Parameters:
        phi: phi
        s3: S_3 observable
        s9: S_9 observable
    
    Returns: d2gamma_P/dq^2dPhi
    '''
    acceptance = 0.5 # placeholder acceptance!!!
    scalar_array = (1/np.pi) * (1 + s3*(2 * np.cos(phi)**2 - 1) + s9*2*np.sin(phi)*np.cos(phi)) * acceptance
    normalized_scalar_array = scalar_array * 2 # normalizes the pdf
    return np.abs(normalized_scalar_array) # returning absolute value to prevent negative values in the log-likelihood sum

def pdf_S(cos_theta_l, cos_theta_k, phi, *params):
    '''
    The pdf across all angular axes in terms of a single S observable.
    Angles should be remapped accordingly.

    Parameters:
        cos_theta_l : remapped cos(theta_l)
        cos_theta_k : remapped cos(theta_k)
        phi : remapped phi angle
        *params : S_i, F_l, A_T2 in that order, where i specifies the index of the S observable

    Returns: d4(gamma + gamma_bar)/...

    '''
    ctl = cos_theta_l
    ctk = cos_theta_k
    c2tl = np.sqrt((ctl**2 + 1) / 2) # cos(2theta_l)
    stl2 = 1-(ctl**2) # sin^2(theta_l)
    stl = np.sqrt(stl2) # sin(theta_l)
    stk2 = 1-(ctk**2) # sin^2(theta_k)
    s2tl = 2*ctl*stl # sin(2theta_l)
    s2tk = 2*ctk*np.sqrt(stk2) # sin(2theta_k)
    
    S_i, F_l, A_T2 = params
    acceptance = 0.5
    Fl_term1 = (3/4) * (1 - F_l) * stk2
    Fl_term2 = F_l * ctk**2
    Fl_term3 = (1/4) * (1 - F_l) * stk2 * c2tl
    Fl_term4 = - F_l * ctk**2 * c2tl
    A_T2_term = (1/2) * (1 - F_l) * A_T2 * stk2 * stl2 * np.cos(2*phi)
    S_i_term = S_i * s2tk * s2tl * np.cos(phi)
    
    scalar_array = (9/(8*np.pi)) * (Fl_term1 + Fl_term2 + Fl_term3 + Fl_term4 + A_T2_term + S_i_term)
    scalar_array *= acceptance
    scalar_array *= 2 # normalization after acceptance
    
    return np.abs(scalar_array)
    

#functions for wrapping angles to input into pdf_S and finding the observables S4 through S8

def S4_angles(cos_theta_l, cos_theta_k, phi):
    'Function for remapping angles to find S4 observable using pdf_S function'
    #copy arrays due to object identity issues
    cos_theta_l = copy.deepcopy(cos_theta_l)
    cos_theta_k = copy.deepcopy(cos_theta_k)
    phi = copy.deepcopy(phi)
    
    #angle transformations
    phi = np.abs(phi) # phi -> -phi where phi < 0
    phi[cos_theta_l < 0] = np.pi - phi[cos_theta_l < 0] # phi -> pi - phi where cos(theta_l) < 0
    cos_theta_l[cos_theta_l < 0] *= -1 # theta_l -> pi - theta_l where theta_l > pi/2
    
    return cos_theta_l, cos_theta_k, phi

def S5_angles(cos_theta_l, cos_theta_k, phi):
    'Function for remapping angles to find S5 observable using pdf_S function'
    #copy arrays due to possible object identity issues
    cos_theta_l = copy.deepcopy(cos_theta_l)
    cos_theta_k = copy.deepcopy(cos_theta_k)
    phi = copy.deepcopy(phi)
    
    #angle transformations
    phi = np.abs(phi) # phi -> -phi where phi < 0
    cos_theta_l[cos_theta_l < 0] *= -1 # theta_l -> pi - theta_l where theta_l > pi/2
    
    return cos_theta_l, cos_theta_k, phi

def S7_angles(cos_theta_l, cos_theta_k, phi):
    'Function for remapping angles to find S7 observable using pdf_S function'
    #copy arrays due to possible object identity issues
    cos_theta_l = copy.deepcopy(cos_theta_l)
    cos_theta_k = copy.deepcopy(cos_theta_k)
    phi = copy.deepcopy(phi)
    
    #angle transformations
    phi[phi > np.pi/2] = np.pi - phi[phi > np.pi/2] # phi -> pi - phi where phi > pi/2
    phi[phi < -np.pi/2] = -np.pi - phi[phi < -np.pi/2] # phi -> -pi - phi where phi < -pi/2
    cos_theta_l[cos_theta_l < 0] *= -1 # theta_l -> pi - theta_l where theta_l > pi/2
    
    return cos_theta_l, cos_theta_k, phi

def S8_angles(cos_theta_l, cos_theta_k, phi):
    'Function for remapping angles to find S8 observable using pdf_S function'
    #copy arrays due to possible object identity issues
    cos_theta_l = copy.deepcopy(cos_theta_l)
    cos_theta_k = copy.deepcopy(cos_theta_k)
    phi = copy.deepcopy(phi)
    
    #angle transformations
    phi[phi > np.pi/2] = np.pi - phi[phi > np.pi/2] # phi -> pi - phi where phi > pi/2
    phi[phi < -np.pi/2] = -np.pi - phi[phi < -np.pi/2] # phi -> -pi - phi where phi < -pi/2
    cos_theta_k[cos_theta_l < 0] *= -1 # theta_k -> pi - theta_k where theta_l > pi/2
    cos_theta_l[cos_theta_l < 0] *= -1 # theta_l -> pi - theta_l where theta_l > pi/2
    
    return cos_theta_l, cos_theta_k, phi
    
    
#log-likelihood functions to be minimized

def log_likelihood(pdf, afb=0, fl=0, s3=0, s9=0, _bin=0):
    """
    The negative log-likelihood of the projected probability distributions defined above
    
    Parameters:
        pdf (func): the probability density function object to be used in the fit
        fl (float): f_l observable
        afb (float): a_fb observable
        s3 (flaot): s3 observable
        s9 (float): s9 observable
        _bin (int): number of the bin to fit
    
    Returns: log likelihood of fitted function
    """
    
    if pdf == pdf_ctl:
        angular_data = bins[_bin]['ctl']
    elif pdf == pdf_ctk:
        angular_data = bins[_bin]['ctk']
    elif pdf == pdf_phi:
        angular_data = bins[_bin]['phi']
    
    normalized_scalar_array = pdf(angular_data, fl=fl, afb=afb, s3=s3, s9=s9)
    return - np.sum(np.log(normalized_scalar_array))

def log_likelihood_S(S_index, _bin, *params):
    """
    The negative log-likelihood of the probability distribution for the S_i observables
    
    Parameters:
        S_index (int): the index of the S_i observable
        *params: the parameters to pass to the pdf_S function
        
    Returns: log likelihood of fitted function
    """
    
    ctl = np.array(bins[_bin]['ctl'])
    ctk = np.array(bins[_bin]['ctk'])
    phi = np.array(bins[_bin]['phi'])
    
    
    if type(S_index) != int:
        raise TypeError('index of S observable must be an integer of value 4, 5, 7, or 8')
    
    if S_index == 4:
        ctl, ctk, phi = S4_angles(ctl, ctk, phi)
    elif S_index == 5:
        ctl, ctk, phi = S5_angles(ctl, ctk, phi)
    elif S_index == 7:
        ctl, ctk, phi = S7_angles(ctl, ctk, phi)
    elif S_index == 8:
        ctl, ctk, phi = S8_angles(ctl, ctk, phi)
        
    else:
        raise ValueError('index of S observable must be an integer of value, 4, 5, 7, or 8')
    
    normalized_scalar_array = pdf_S(ctl, ctk, phi, *params)
    return -np.sum(np.log(normalized_scalar_array))

# function for minimizing the log likelihood

def minimize_logL(pdf, _bin, afb=0, fl=0, s3=0, s4=0, s5=0, s7=0, s8=0, s9=0, S_index = False):
    '''
    Uses the iminuit library to minimize the negated log-likelihood function
    for a probability density function in parameter space to establish a fit

    Parameters:
        pdf: the probability density function used for the fit
        _bin: which q^2 bin to fit the observable values for
        S_index: if using pdf_S, set the index of the observable here
        fl: the fl value if using pdf_S, otherwise an initial guess for fl
        other: an initial guess of the observable parameters
    
    Returns: minimized iminuit.Minuit object
    '''
    
    if type(pdf) != type(pdf_ctl):
        raise TypeError('pdf must be a function object')
        
        
    elif pdf == pdf_ctl:
        min_func = lambda afb, fl:log_likelihood(pdf, afb=afb, fl=fl, _bin=_bin) # uses a placeholder lambda function of log-likelihood so that the minimizer only minimizes wrt correct observables 
        min_func.errordef = iminuit.Minuit.LIKELIHOOD
        m = iminuit.Minuit(min_func, afb=afb, fl=fl)
        m.limits=((-1, 1), (-1, .9)) # these limits are good for the Jupyter Notebook toy data - however they have an effect on convergence and may need to be changed

    
    elif pdf == pdf_ctk:
        min_func = lambda fl:log_likelihood(pdf, fl=fl, _bin=_bin)
        min_func.errordef = iminuit.Minuit.LIKELIHOOD
        m = iminuit.Minuit(min_func, fl=fl)
    
    elif pdf == pdf_phi:
        min_func = lambda s3, s9:log_likelihood(pdf, s3=s3, s9=s9, _bin=_bin)
        min_func.errordef = iminuit.Minuit.LIKELIHOOD
        m = iminuit.Minuit(min_func, s3=s3, s9=s9)
        #todo: find good boundaries on s3 and s9 for the values returned to make sense
        # maybe try a contour plot
        #todo has been done
    
    elif S_index: #abuses dynamic typing - a non-zero integer will evaluate to True
        #this will require putting in a known value of fl into the function
        min_func = lambda S_i, A_T2:log_likelihood_S(S_index, _bin, S_i, fl, A_T2)
        min_func.errordef = iminuit.Minuit.LIKELIHOOD
        m = iminuit.Minuit(min_func, S_i = 0, A_T2 = 0) # kinda just letting 0 be the initial guess for each
        m.limits = ((-1, 1), (-1, 1))
    
    else:
        raise NotImplementedError('This pdf is not yet defined')
        
        
    m.migrad()
    m.hesse()
    return m


def toy_data_observables():
    '''
    A function to calculate all observables from the projected pdf fits for the toy data
    
    Returns the values of the observables and their errors as two separate arrays
    The order of the bins is preserved in the array
    '''
    values = []
    errors = []
    for i, _bin in enumerate(bins):
        m_ctl = minimize_logL(pdf_ctl, i)
        #print(f'ctl minimum is valid: {m_ctl.fmin.is_valid}')
        m_ctk = minimize_logL(pdf_ctk, i, fl = .5) # initial guess is important here
        #print(f'ctk minimum is valid: {m_ctk.fmin.is_valid}')
        m_phi = minimize_logL(pdf_phi, i)
        #print(f'phi minimum is valid: {m_phi.fmin.is_valid}')
        
        m_S4 = minimize_logL(pdf_S, i, fl=m_ctl.values[1], S_index = 4)
        #print(f'bin {i} S4 minimum is valid: {m_S4.fmin.is_valid}')
        m_S5 = minimize_logL(pdf_S, i, fl=m_ctl.values[1], S_index = 5)
        #print(f'bin {i} S5 minimum is valid: {m_S5.fmin.is_valid}')
        m_S7 = minimize_logL(pdf_S, i, fl=m_ctl.values[1], S_index = 7)
        #print(f'bin {i} S7 minimum is valid: {m_S7.fmin.is_valid}')
        m_S8 = minimize_logL(pdf_S, i, fl=m_ctl.values[1], S_index = 8)
        #print(f'bin {i} S8 minimum is valid: {m_S8.fmin.is_valid}')
        
        afb_val = m_ctl.values[0]
        afb_err = m_ctl.errors[0]
        fl_ctl_val = m_ctl.values[1]
        fl_ctl_err = m_ctl.errors[1]
        
        fl_ctk_val = m_ctk.values[0]
        fl_ctk_err = m_ctk.errors[0]
        
        s3_val = m_phi.values[0]
        s3_err = m_phi.errors[0]
        s9_val = m_phi.values[1]
        s9_err = m_phi.errors[1]
        
        s4_val = m_S4.values[0]
        s4_err = m_S4.errors[0]
        
        s5_val = m_S5.values[0]
        s5_err = m_S5.errors[0]
        
        s7_val = m_S7.values[0]
        s7_err = m_S7.errors[0]
        
        s8_val = m_S8.values[0]
        s8_err = m_S8.errors[0]
        
        values.append([afb_val, fl_ctl_val, fl_ctk_val, s3_val, s4_val, s5_val, s7_val, s8_val, s9_val])
        errors.append([afb_err, fl_ctl_err, fl_ctk_err, s3_err, s4_err, s5_err, s7_err, s8_err, s9_err])
    
    return np.array(values), np.array(errors)



#code to generate and save toy data observable values and errors

vals, errs = toy_data_observables()
vals = pd.DataFrame(vals)
errs = pd.DataFrame(errs)

#labels = np.array(['afb', 'fl from ctl', 'fl from ctk', 's3', 's4', 's5', 's7', 's8', 's9'])
#vals.to_csv('toy_data_observable_values.csv', header = labels, index = True)
#errs.to_csv('toy_data_observable_errors.csv', header = labels, index = True)
