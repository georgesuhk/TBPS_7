# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:50:43 2023

@author: victo
"""

import numpy as np
import matplotlib.pyplot as plt
import fitting_functions as fit
from matplotlib import rcParams

rcParams.update({'font.size':22})

#some contour visualization functions for visually finding minima

def contour_plot(x_range, y_range, func):
    '''
    Produces a contour plot for a function with 2D input and 1D output in the ranges x_range and y_range
    
    Parameters:
        x_range: list/tuple of length two; the range of the observable to plot on x-axis
        y_range: list/tuple of length two; the range of the observable to plot on y-axis
        func: function object in form f(x, y); the function to which the observables are passed to create the heightmap
    '''
    x_length = 100 # side length of the array
    y_length = 100
    xs = np.linspace(x_range[0], x_range[1], x_length)
    ys = np.linspace(y_range[0], y_range[1], y_length)
    x_coord_array = np.broadcast_to(xs, [y_length, x_length]) # a 2D array where each entry describes its own x coordinate
    y_coord_array = np.broadcast_to(ys, [y_length, x_length]).T # a 2D array where each entry describes its own y coordinate
    
    heightmap = np.zeros([y_length, x_length])
    
    for y_index, y_val in enumerate(ys):
        for x_index, x_val in enumerate(xs):
            heightmap[y_index][x_index] = func(x_val, y_val)
        
    contours = plt.contour(x_coord_array, y_coord_array, heightmap)
    for i, label in enumerate(contours.cvalues):
        contours.collections[i].set_label(label)
    plt.legend(title = r'$log(\mathcal{L})$ contour values')

def toy_data_contours(_bin):
    'Creates plots to visualize the minimization of each projected pdf in a specific bin'
    
    ctl = lambda afb, fl:fit.log_likelihood(fit.projection_ctl, afb=afb, fl=fl, _bin=_bin)
    ctk = lambda fl:fit.log_likelihood(fit.projection_ctk, fl=fl, _bin=_bin)
    phi = lambda s3, s9:fit.log_likelihood(fit.projection_phi, s3=s3, s9=s9, _bin=_bin)
    
    #playing around with these values gives some intuition for the shape of the function
    ctl_range_x = [-.4, .2] # range in afb
    ctl_range_y = [-.5, 1] # range in fl for ctl
    ctk_range = [-.5, 1] # range in fl for ctk
    phi_range_x = [-1.5, .5] # range in s3
    phi_range_y = [-1.5, -.5] # range in s9
    
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.set_title(r'$log\left(\mathcal{L}\left(\frac{d^{2}\Gamma_P}{dq^{2}dcos(\theta_l)}\right)\right)$, bin %.i' %_bin)
    ax1.set_xlabel(r'$A_{fb}$')
    ax1.set_ylabel(r'$F_{l}$')
    contour_plot(ctl_range_x, ctl_range_y, ctl)
    
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.set_title(r'$log\left(\mathcal{L}\left(\frac{d^{2}\Gamma_P}{dq^{2}d\Phi}\right)\right)$, bin %.i' %_bin)
    ax2.set_xlabel(r'$S_3$')
    ax2.set_ylabel(r'$S_9$')
    contour_plot(phi_range_x, phi_range_y, phi)
    
    fig3 = plt.figure()
    ax3 = plt.axes()
    ax3.set_title(r'Minimization for $cos(\theta_k)$, bin %.i' %_bin)
    ax3.set_xlabel(r'$F_l$')
    ax3.set_ylabel(r'$log\left(\mathcal{L}\left(\frac{d^{2}\Gamma_P}{dq^{2}dcos(\theta_k)}\right)\right)$')
    ctks = np.linspace(ctk_range[0], ctk_range[1], 50)
    output = np.array([ctk(val) for val in ctks])
    plt.plot(ctks, output)
    
    