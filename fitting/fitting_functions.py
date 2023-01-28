# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 12:29:45 2023

@author: victor
"""

import numpy as np
import pandas as pd
import iminuit

data_path = 'C:/Users/victo/ic-teach-kstmumu-public/kstarmumu_toy_data/'
#remember to have the correct path for this part
#currently accessing toy data
files = [f'{data_path}toy_data_bin_{i}.csv' for i in range(7)]
bins = [pd.read_csv(file) for file in files]



#Functions defined for fitting


    
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
    return normalized_scalar_array


def pdf_ctl(cos_theta_l, afb=0, fl=0, s3=0, s9=0):
    """
    The projection of the pdf onto cos(theta_l)
    
    Only parameters relevant to the function are listed in the docstring;
    this is for a general log-likelihood function
    Parameters:
        cos_theta_l: cos(theta_l)
        fl: f_l observable
        afb: a_fb observable
    
    Returns: d2gamma_P/dq^2dcos(theta_l)
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    acceptance = 0.5  # acceptance "function"
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance
    normalized_scalar_array = scalar_array * 2 # normalising scalar array to account for the non-unity acceptance function
    return normalized_scalar_array

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
    return normalized_scalar_array


    
from scipy.special import erf

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
    
    A = (n/np.abs(alpha))**n * np.exp(-(np.abs(alpha))**2 / 2)
    B = (n/np.abs(alpha)) - np.abs(alpha)
    C = (n/np.abs(alpha)) * (1/(n-1)) * np.exp(-(np.abs(alpha))**2 / 2)
    D = np.sqrt(np.pi/2) * (1 + erf(np.abs(alpha))/np.sqrt(2))
    N = 1/(sigma * (C + D))
    
    
    bool_arr = ((x-mean)/sigma > -alpha)
    gaussian_arr = bool_arr * N * np.exp(-((x-mean)**2)/(2*sigma**2))
    power_law_arr = (bool_arr == False) * A * (B - (x-mean)/sigma)**(-n)
    
    return gaussian_arr + power_law_arr
    
    
def mB_dist(x, mean, alpha1, alpha2, n1, n2, sigma1, sigma2):
    return crystal_ball(x, mean, alpha1, n1, sigma1) + crystal_ball(x, mean, alpha2, n2, sigma2)   

#log-likelihood function to be minimized

def log_likelihood(pdf, afb=0, fl=0, s3=0, s9=0, _bin=0):
    """
    The negative log-likelihood of the probability distributions defined above
    
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


# function for minimizing the log likelihood

def minimize_logL(pdf, _bin, afb=0, fl=0, s3=0, s9=0):
    '''
    Uses the iminuit library to minimize the negated log-likelihood function
    for a projected angular probability density function

    Parameters:
        pdf: the probability density function used for the fit
        _bin: which q^2 bin to fit the observable values for
        starting_vals: an initial guess of the observable parameters, in the same positional order as for the function
    
    Returns: minimized iminuit.Minuit object
    '''
    
    if type(pdf) != type(pdf_ctl):
        raise TypeError('pdf must be a function object')
        
        
    elif pdf == pdf_ctl:
        min_func = lambda afb, fl:log_likelihood(pdf, afb=afb, fl=fl, _bin=_bin) # uses a placeholder lambda function of log-likelihood so that the minimizer only minimizes wrt correct observables 
        min_func.errordef = iminuit.Minuit.LIKELIHOOD
        m = iminuit.Minuit(min_func, afb=afb, fl=fl)
        m.limits=((-1, 1), (-1, .9)) # these limits are good for the Jupyter Notebook data - however they have an effect on convergence and may need to be changed

    
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
    
    else:
        raise NotImplementedError('This pdf is not yet defined')
        
        
    m.migrad()
    m.hesse()
    return m


def toy_data_observables():
    '''
    A function to calculate all observables from the projection for the toy data
    
    Returns the values of the observables and their errors as two separate arrays
    The order of the bins is preserved in the array
    '''
    values = []
    errors = []
    for i, _bin in enumerate(bins):
        m_ctl = minimize_logL(pdf_ctl, i)
        m_ctk = minimize_logL(pdf_ctk, i)
        m_phi = minimize_logL(pdf_phi, i)
        
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
        
        values.append([afb_val, fl_ctl_val, fl_ctk_val, s3_val, s9_val])
        errors.append([afb_err, fl_ctl_err, fl_ctk_err, s3_err, s9_err])
    
    return np.array(values), np.array(errors)




#vals, errs = toy_data_observables()
#vals = pd.DataFrame(vals)
#errs = pd.DataFrame(errs)

#labels = np.array(['afb', 'fl_ctl', 'fl_ctk', 's3', 's9'])
#vals.to_csv('toy_data_observables.csv', header = labels, index = False)
#vals.to_csv('toy_data_errors.csv', header = labels, index = False)
        
        



