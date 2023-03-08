#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:39:15 2023

@author: moriahsanusi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import iminuit
#import Efficiency_legendre_fit as acc
#import Fitting_visualisation2 as fv
import multiD_fitting as mdf
import particle_antiparticle_acceptance as paa

data = pd.read_csv('cleaned_total_dataset.csv') #importing dataset

def q2_binning_sm(data, bin_ranges):
    return [data[(data['q2'] >= bin_range[0]) & (data['q2'] <= bin_range[1])] for bin_range in bin_ranges]



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

b0_id = data['B0_ID'] #selecting the B0_id column

#particle = []
#anti_particles = []
"""
for i in b0_id:
    if i == 511:
        particles = data.iloc[i]
    
    else:
        antiparticles = data.iloc[i]
        
        
"""

particles = data[data['B0_ID']==511]
antiparticles = data[data['B0_ID']==-511]

particles.to_csv('Particles.csv', index = True)
antiparticles.to_csv('Antiparticles.csv', index = True)

particles = data[data['B0_ID']==511]
antiparticles = data[data['B0_ID']==-511]

bin_particles = q2_binning_sm(particles, bin_ranges)
bin_antiparticles = q2_binning_sm(antiparticles, bin_ranges)

bins = q2_binning_sm(particles, bin_ranges)
#%%
#PARTICLES

def particle_asym_func(cos_theta_l, cos_theta_k, phi, _bin, *params):
    """
    

    Parameters
    ----------
    cos_theta_l : arr
        array of cos(theta_l) for particle candidates
    cos_theta_k : arr
        array of cos(theta_k) for particle candidates
    phi : array of phi for particle candidates 
        DESCRIPTION.
    _bin : 
        bins 
    *params : I1s, I1c, I2s, I2c, I3, I4, I5, I6, I7, I8, I9 in that order 
        

    Returns
    -------
    scalar_array : an array of values for the expression visible in the OneNote
        

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
    
    I1s, I1c, I2s, I2c, I3, I4, I5, I6, I7, I8, I9 = params
    
    q2 = np.array([np.mean(bin_ranges[_bin]) for n in range(len(ctl))]).astype('float64')
    acceptance = paa.acceptance_particle(ctl, ctk, phi, q2)
    print(acceptance)
    
    term1 = I1s*stk2 
    term2 = I1c*(ctk**2)
    term3 = I2s*stk2*c2tl
    term4 = I2c*(ctk**2)*c2tl
    term5 = I3 * stk2 * stl2 * np.cos(2 * phi)
    term6 = I4 * s2tk * s2tl * np.cos(phi)
    term7 = I5 * s2tk * stl * np.cos(phi)
    term8 = I6 * stk2 * ctl
    term9 = I7 * s2tk * stl * np.sin(phi)
    term10 = I8 * s2tk * s2tl * np.sin(phi)
    term11 = I9 * stk2 * stl2 * np.sin(2 * phi)
    
    scalar_array = (9/(32*np.pi)) * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11)
    scalar_array *= acceptance   
    
    return scalar_array
    


def particle_log_likelihood(_bin, *params):
    '''
    The negative log-likelihood of the probability distribution of the I_i observables defined above with given parameters
    
    Parameters
    ----------
    *params: I1s, I1c, I2s, I2c, I3, I4, I5, I6, I7, I8, I9 in that order 
    
    Returns: log likelihood of function with those parameters
    '''
    ctl = bin_particles[_bin]['costhetal']
    ctk = bin_particles[_bin]['costhetak']
    phi = bin_particles[_bin]['phi']
    
    #limiting the value of ctk for testing purposes
    
    normalized_scalar_array = particle_asym_func(ctl, ctk, phi, _bin, *params)
    if np.any(normalized_scalar_array < 0):
        return 1e6 * np.sum(normalized_scalar_array < 0) # punish it by how many negative values come out - might be to flat since this will be discrete
    return - np.sum(np.log(normalized_scalar_array))



def particle_minimize_logL(_bin, initial_guess):
    '''
    Minimizes the log-likelihood for the full I_i observables pdf across all angles

    Parameters
    ----------
    _bin : the bin to fit to
    initial_guess : the initial guess of parameter values
    I1s, I1c, I2s, I2c, I3, I4, I5, I6, I7, I8, I9 in that order 

    Returns
    -------
    m : minimized iminuit.Minuit object

    '''
    min_func = lambda *args:particle_log_likelihood(_bin, *args)
    min_func.errordef = iminuit.Minuit.LIKELIHOOD
    m = iminuit.Minuit(min_func, *initial_guess)
    m.limits = [(0., 0.375), (0.,0.5 ), (0., 0.125), (-0.5, 0.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.)]
    #m.fixed['x0'] = True
    m.migrad()
    m.hesse()
    
    return m  

def particle_get_observables(bin_number=len(bin_ranges)):
    """
    

    Parameters
    ----------
    bin_number : TYPE, optional
        DESCRIPTION. The default is len(bin_ranges).

    Returns
    -------
    Two arrays- array of the I_i observables and an array of the errors of the
    I_i observables for the particles only

    """
    
    vals = []
    errs = []
    for i in range(bin_number):
        m = particle_minimize_logL(i, [0.2, 0.2, 0.2, 0.2, 0., 0., 0., 0., 0., 0., 0.])
        bin_vals = list(m.values)
        print(f'bin {i} minimum is valid: {m.fmin.is_valid}')
        bin_errs = list(m.errors)
        vals.append(bin_vals)
        errs.append(bin_errs)
    return np.array(vals), np.array(errs)


p_labels =  ['I1s', 'I1c', 'I2s', 'I2c', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9']
p_err_labels =  ['I1s_err', 'I1c_err', 'I2s_err', 'I2c_err', 'I3_err', 'I4_err', 'I5_err', 'I6_err', 'I7_err', 'I8_err', 'I9_err']

particles_vals, particles_errs = particle_get_observables()

p_err_df = pd.DataFrame(particles_errs, columns= p_err_labels)
p_df =  pd.DataFrame(particles_vals, columns= p_labels)

p_df.to_csv('particles_vals.csv', index=True)
p_err_df.to_csv('particles_errs.csv', index = True)




#%%%
#antiparticles 

def antiparticle_asym_func(cos_theta_l, cos_theta_k, phi, _bin, *params):
    """
    

    Parameters
    ----------
    cos_theta_l : arr
        array of cos(theta_l) for antiparticle candidates
    cos_theta_k : arr
        array of cos(theta_k) for antiparticle candidates
    phi : array of phi for antiparticle candidates 
        DESCRIPTION.
    _bin : 
        bins 
    *params : I1s, I1c, I2s, I2c, I3, I4, I5, I6, I7, I8, I9 in that order 
        

    Returns
    -------
    scalar_array : an array of values for the expression visible in the OneNote

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
    
    I1s, I1c, I2s, I2c, I3, I4, I5, I6, I7, I8, I9 = params
    
    q2 = np.array([np.mean(bin_ranges[_bin]) for n in range(len(ctl))]).astype('float64')
    acceptance = paa.acceptance_antiparticle(ctl, ctk, phi, q2)
    
    term1 = I1s*stk2 
    term2 = I1c*(ctk**2)
    term3 = I2s*stk2*c2tl
    term4 = I2c*(ctk**2)*c2tl
    term5 = I3 * stk2 * stl2 * np.cos(2 * phi)
    term6 = I4 * s2tk * s2tl * np.cos(phi)
    term7 = I5 * s2tk * stl * np.cos(phi)
    term8 = I6 * stk2 * ctl
    term9 = I7 * s2tk * stl * np.sin(phi)
    term10 = I8 * s2tk * s2tl * np.sin(phi)
    term11 = I9 * stk2 * stl2 * np.sin(2 * phi)
    
    scalar_array = (9/(32*np.pi)) * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11)
    scalar_array *= acceptance   
    
    return scalar_array
    


def antiparticle_log_likelihood(_bin, *params):
    '''
    The negative log-likelihood of the probability distribution of the I_i observables defined above with given parameters
    
    Parameters
    ----------
    *params: I1s, I1c, I2s, I2c, I3, I4, I5, I6, I7, I8, I9 in that order 
    
    Returns: log likelihood of function with those parameters
    '''
    ctl = bin_antiparticles[_bin]['costhetal']
    ctk = bin_antiparticles[_bin]['costhetak']
    phi = bin_antiparticles[_bin]['phi']
    
    #limiting the value of ctk for testing purposes
    
    normalized_scalar_array = antiparticle_asym_func(ctl, ctk, phi, _bin, *params)
    if np.any(normalized_scalar_array < 0):
        return 1e6 * np.sum(normalized_scalar_array < 0) # punish it by how many negative values come out - might be to flat since this will be discrete
    return - np.sum(np.log(normalized_scalar_array))



def antiparticle_minimize_logL(_bin, initial_guess):
    '''
    Minimizes the log-likelihood for the full I_i observables pdf across all angles

    Parameters
    ----------
    _bin : the bin to fit to
    initial_guess : the initial guess of parameter values
    I1s, I1c, I2s, I2c, I3, I4, I5, I6, I7, I8, I9 in that order 

    Returns
    -------
    m : minimized iminuit.Minuit object

    '''
    min_func = lambda *args:antiparticle_log_likelihood(_bin, *args)
    min_func.errordef = iminuit.Minuit.LIKELIHOOD
    m = iminuit.Minuit(min_func, *initial_guess)
    m.limits = [(0., 0.375), (0.,0.5 ), (0., 0.125), (-0.5, 0.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.)]
    #m.fixed['x0'] = True
    m.migrad()
    m.hesse()
    
    return m  

def antiparticle_get_observables(bin_number=len(bin_ranges)):
    """ 
    Parameters
    ----------
    bin_number : TYPE, optional
        DESCRIPTION. The default is len(bin_ranges).

    Returns
    -------
    Two arrays- array of the I_i observables and an array of the errors of the
    I_i observables for the antiparticles
    """
    
    vals = []
    errs = []
    for i in range(bin_number):
        m = antiparticle_minimize_logL(i, [0.2, 0.2, 0.2, 0.2, 0., 0., 0., 0., 0., 0., 0.])
        bin_vals = list(m.values)
        print(f'bin {i} minimum is valid: {m.fmin.is_valid}')
        bin_errs = list(m.errors)
        vals.append(bin_vals)
        errs.append(bin_errs)
    return np.array(vals), np.array(errs)



ap_labels =  ['I1sbar', 'I1cbar', 'I2sbar', 'I2cbar', 'I3bar', 'I4bar', 'I5bar', 'I6bar', 'I7bar', 'I8bar', 'I9bar']
ap_err_labels =  ['I1sbar_err', 'I1cbar_err', 'I2sbar_err', 'I2cbar_err', 'I3bar_err', 'I4bar_err', 'I5bar_err', 'I6bar_err', 'I7bar_err', 'I8bar_err', 'I9bar_err']

antiparticles_vals, antiparticles_errs = antiparticle_get_observables()

ap_err_df = pd.DataFrame(antiparticles_errs, columns= ap_err_labels)
ap_df =  pd.DataFrame(antiparticles_vals, columns= ap_labels)

ap_df.to_csv('antiparticles_vals.csv', index=True)
ap_err_df.to_csv('antiparticles_errs.csv', index = True)

#%%

#calculating the A's


I3 = p_df['I3']
I4 = p_df['I4']
I5 = p_df['I5']
I6 = p_df['I6']
I7 = p_df['I7']
I8 = p_df['I8']
I9 = p_df['I9']

I3bar = ap_df['I3bar']
I4bar = ap_df['I4bar']
I5bar = ap_df['I5bar']
I6bar = ap_df['I6bar']
I7bar = ap_df['I7bar']
I8bar = ap_df['I8bar']
I9bar = ap_df['I9bar']

I3_err = p_err_df['I3_err']
I4_err = p_err_df['I4_err']
I5_err = p_err_df['I5_err']
I6_err = p_err_df['I6_err']
I7_err = p_err_df['I7_err']
I8_err = p_err_df['I8_err']
I9_err = p_err_df['I9_err']

I3bar_err = ap_err_df['I3bar_err']
I4bar_err = ap_err_df['I4bar_err']
I5bar_err = ap_err_df['I5bar_err']
I6bar_err = ap_err_df['I6bar_err']
I7bar_err = ap_err_df['I7bar_err']
I8bar_err = ap_err_df['I8bar_err']
I9bar_err = ap_err_df['I9bar_err']

A3 = I3 - I3bar 
A4 = I4 - I4bar
A5 = I5 - I5bar
A6 = I6 - I6bar
A7 = I7 - I7bar 
A8 = I8 - I8bar
A9 = I9 - I9bar 


#calculating the A errors 

A3_err = np.sqrt((I3_err **2) + (I3bar_err **2))
A4_err = np.sqrt((I4_err **2) + (I4bar_err **2))
A5_err = np.sqrt((I5_err **2) + (I5bar_err **2))
A6_err = np.sqrt((I6_err **2) + (I6bar_err **2))
A7_err = np.sqrt((I7_err **2) + (I7bar_err **2))
A8_err = np.sqrt((I8_err **2) + (I8bar_err **2))
A9_err = np.sqrt((I9_err **2) + (I9bar_err **2))


"""
#fv.plot_real_observables(A3, A3_err, bin_ranges)
bin_centers = np.array([(bin_range[1] + bin_range[0])/2 for bin_range in bin_ranges])
bin_widths = np.array([(bin_range[1] - bin_range[0])/2 for bin_range in bin_ranges])
plt.plot(bin_centers, A3, 'x')
plt.errorbar(bin_centers, A3, xerr = bin_widths, yerr = A3_err, fmt = 'o', capsize = 4, capthick=2, color = 'xkcd:coral', ecolor = 'xkcd:cobalt', elinewidth=2)
"""

A_array = np.column_stack((A3, A4, A5, A6, A7, A8, A9))
#A_array = np.array(A_list)

A_err_array = np.column_stack((A3_err, A4_err, A5_err, A6_err, A7_err, A8_err, A9_err))


#plt.plot(bin_centers, A_array,'x')



def plot_A(vals, errs, bin_ranges):
    fig, ax = plt.subplots(2, 4)
    ax[0][0].set_title(r'$A3$', fontsize = 10)
    ax[0][1].set_title(r'$A4$', fontsize = 10)
    ax[0][2].set_title(r'$A5$', fontsize = 10)
    ax[0][3].set_title(r'$A6$', fontsize = 10)
    ax[1][0].set_title(r'$A7$', fontsize = 10)
    ax[1][1].set_title(r'$A8$', fontsize = 10)
    ax[1][2].set_title(r'$A9$', fontsize = 10)
    ax[1][3].set_title(r'$P{\prime}_8$', fontsize = 10)
    fig.delaxes(ax[1][3])
    #fig.delaxes(ax[1][2])
    bin_centers = np.array([(bin_range[1] + bin_range[0])/2 for bin_range in bin_ranges])
    bin_widths = np.array([(bin_range[1] - bin_range[0])/2 for bin_range in bin_ranges])
    #plot_predictions('p_pred.csv', bin_ranges, ax)
    for i in range(7):
        ax[int(i/4)][int(i%4)].errorbar(bin_centers, vals[:,i], xerr = bin_widths, yerr = errs[:,i], fmt = 'o', capsize = 4, capthick=2, color = 'xkcd:coral', ecolor = 'xkcd:cobalt', elinewidth=2)
        ax[int(i/4)][int(i%4)].set_xlabel(r'$q^2$', fontsize=8)
        ax[int(i/4)][int(i%4)].tick_params(axis = 'both', labelsize  = 8)
        ax[int(i/4)][int(i%4)].grid()
        
        plt.savefig("A_vals.png")

#plot_A(A_array, A_err_array, bin_ranges)
#%%

#calculating I_i + Ibar_i 

s_dataframe = mdf.s_df
A_fb = mdf.s_df['A_fb']
S3 = mdf.s_df['S3']
S4 = mdf.s_df['S4']
S5 = mdf.s_df['S5']
S7 = mdf.s_df['S7']
S8 = mdf.s_df['S8']
S9 = mdf.s_df['S9']

#errors in angular observables
A_fb_err = mdf.s_err_df['A_fb_err']
S3_err = mdf.s_err_df['S3_err']
S4_err = mdf.s_err_df['S4_err']
S5_err = mdf.s_err_df['S5_err']
S7_err = mdf.s_err_df['S7_err']
S8_err = mdf.s_err_df['S8_err']
S9_err = mdf.s_err_df['S9_err']


nc3 = S3 / (I3 + I3bar)
nc4 = S4 / (I4 + I4bar)
nc5 = S5 / (I5 + I5bar)
nc6 = A_fb / (I6 + I6bar)
nc7 = S7 / (I7 + I7bar)
nc8 = S8 / (I8 + I8bar)
nc9 = S9 / (I9 + I9bar)

A3_norm = A3 * nc3
A4_norm = A4 * nc4
A5_norm = A5 * nc5
A6_norm = A6 * nc6
A7_norm = A7 * nc7
A8_norm = A8 * nc8
A9_norm = A9 * nc9

#calculating the errors 
#calculating the erros in I_i + Ibar_i

nc3_err = nc3 * np.sqrt(((S3_err/S3)**2) + ((A3_err/(I3 + I3bar))**2))
nc4_err = nc4 * np.sqrt(((S4_err/S4)**2) + ((A4_err/(I4 + I4bar))**2))
nc5_err = nc5 * np.sqrt(((S5_err/S5)**2) + ((A5_err/(I5 + I5bar))**2))
nc6_err = nc6 * np.sqrt(((A_fb_err/A_fb)**2) + ((A6_err/(I6 + I6bar))**2))
nc7_err = nc7 * np.sqrt(((S7_err/S7)**2) + ((A7_err/(I7 + I7bar))**2))
nc8_err = nc8 * np.sqrt(((S8_err/S8)**2) + ((A8_err/(I8 + I8bar))**2))
nc9_err = nc9 * np.sqrt(((S9_err/S9)**2) + ((A9_err/(I9 + I9bar))**2))

A3_norm_err = A3_norm * np.sqrt(((A3_err/A3)**2) + ((nc3_err/nc3)**2))
A4_norm_err = A4_norm * np.sqrt(((A4_err/A4)**2) + ((nc4_err/nc4)**2))
A5_norm_err = A5_norm * np.sqrt(((A5_err/A5)**2) + ((nc5_err/nc5)**2))
A6_norm_err = A6_norm * np.sqrt(((A6_err/A6)**2) + ((nc6_err/nc6)**2))
A7_norm_err = A7_norm * np.sqrt(((A7_err/A7)**2) + ((nc7_err/nc7)**2))
A8_norm_err = A8_norm * np.sqrt(((A8_err/A8)**2) + ((nc8_err/nc8)**2))
A9_norm_err = A9_norm * np.sqrt(((A9_err/A9)**2) + ((nc9_err/nc9)**2))


A_norm = np.column_stack((A3_norm, A4_norm, A5_norm, A6_norm, A7_norm, A8_norm, A9_norm))
A_norm_err = np.column_stack((A3_norm_err, A4_norm_err, A5_norm_err, A6_norm_err, A7_norm_err, A8_norm_err, A9_norm_err))

plot_A(A_norm, A_norm_err, bin_ranges)


A_vals_labels = ['A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
A_errs_labels = ['A3_err', 'A4_err', 'A5_err', 'A6_err', 'A7_err', 'A8_err', 'A9_err']

A_vals_df = pd.DataFrame(A_norm, columns = A_vals_labels)
A_errs_df = pd.DataFrame(A_norm_err, columns = A_errs_labels)

A_vals_df.to_csv('A_vals2.csv', index = True)
A_errs_df.to_csv('A_errs2.csv', index = True )

#%%

