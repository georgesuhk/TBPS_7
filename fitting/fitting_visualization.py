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

def plot_toy_data_contours(_bin):
    'Creates plots to visualize the minimization of each projected pdf in a specific bin'
    
    # defining new functions to avoid typing later on
    ctl = lambda afb, fl:fit.log_likelihood(fit.pdf_ctl, afb=afb, fl=fl, _bin=_bin)
    ctk = lambda fl:fit.log_likelihood(fit.pdf_ctk, fl=fl, _bin=_bin)
    phi = lambda s3, s9:fit.log_likelihood(fit.pdf_phi, s3=s3, s9=s9, _bin=_bin)
    
    #playing around with these values gives some intuition for the shape of the function
    ctl_range_x = [-.2, .9] # range in afb
    ctl_range_y = [0, 1.5] # range in fl for ctl
    ctk_range = [-1, 1] # range in fl for ctk
    phi_range_x = [-.2, .2] # range in s3
    phi_range_y = [-.2, .2] # range in s9
    
    #creating the contours for cos(theta_l)
    fig1 = plt.figure()
    ax1 = plt.axes()
    ax1.set_title(r'$log\left(\mathcal{L}\left(\frac{d^{2}\Gamma_P}{dq^{2}dcos(\theta_l)}\right)\right)$, bin %.i' %_bin)
    ax1.set_xlabel(r'$A_{fb}$')
    ax1.set_ylabel(r'$F_{l}$')
    contour_plot(ctl_range_x, ctl_range_y, ctl)
    
    #creating the contours for phi
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.set_title(r'$log\left(\mathcal{L}\left(\frac{d^{2}\Gamma_P}{dq^{2}d\Phi}\right)\right)$, bin %.i' %_bin)
    ax2.set_xlabel(r'$S_3$')
    ax2.set_ylabel(r'$S_9$')
    contour_plot(phi_range_x, phi_range_y, phi)
    
    #creating the plot for cos(theta_k)
    fig3 = plt.figure()
    ax3 = plt.axes()
    ax3.set_title(r'Minimization for $cos(\theta_k)$, bin %.i' %_bin)
    ax3.set_xlabel(r'$F_l$')
    ax3.set_ylabel(r'$log\left(\mathcal{L}\left(\frac{d^{2}\Gamma_P}{dq^{2}dcos(\theta_k)}\right)\right)$')
    ctks = np.linspace(ctk_range[0], ctk_range[1], 50)
    output = np.array([ctk(val) for val in ctks])
    plt.plot(ctks, output)
    
def plot_toy_data_fits(_bin):
    'Fits all projected pdfs to their respective histograms and plots the fitted function for a specified bin'
    
    fig, ax = plt.subplots(1, 3)
    
    # minimizing to find the values of the observables
    m_ctl = fit.minimize_logL(fit.pdf_ctl, afb = -.1, fl = 0.3, _bin = _bin)
    afb = m_ctl.values[0]
    fl_ctl = m_ctl.values[1]
    
    m_ctk = fit.minimize_logL(fit.pdf_ctk, _bin = _bin, fl = 0.2)
    fl_ctk = m_ctk.values[0]
    
    m_phi = fit.minimize_logL(fit.pdf_phi, _bin = _bin, s3=-0.001, s9=-0.001)
    s3 = m_phi.values[0]
    s9 = m_phi.values[1]
    
    #plotting to compare histograms to the pdf fits
    ctls = fit.bins[_bin]['ctl']
    heights, edges, patches = ax[0].hist(ctls, color = 'xkcd:powder blue')
    centers = 0.5*(edges[1]-edges[0]) + edges[:-1]
    display_x = np.linspace(edges.min(), edges.max(), 50) # xs to fill range of plot
    true_x = np.linspace(centers.min(), centers.max(), 50) # xs for calculating area
    display_y = fit.pdf_ctl(display_x, afb = afb, fl = fl_ctl) # ys to match plotting xs
    true_y = fit.pdf_ctl(true_x, afb = afb, fl = fl_ctl) # ys to match calculating xs
    scale = np.trapz(heights, centers) / np.trapz(true_y, true_x) # normalize distribution to histogram using numpy numerical integration
    
    ax[0].plot(display_x, display_y*scale, color = 'red', linewidth = 3)
    ax[0].set_xlabel(r'$cos(\theta_l)$')
    ax[0].set_ylabel(r'Candidates')
    
    ctks = fit.bins[_bin]['ctk']
    heights, edges, patches = ax[1].hist(ctks, color = 'xkcd:powder blue')
    centers = 0.5*(edges[1]-edges[0]) + edges[:-1]
    display_x = np.linspace(edges.min(), edges.max(), 50) # xs to fill range of plot
    true_x = np.linspace(centers.min(), centers.max(), 50) # xs for calculating area
    display_y = fit.pdf_ctk(display_x, fl = fl_ctk) # ys to match plotting xs
    true_y = fit.pdf_ctk(true_x, fl = fl_ctk) # ys to match calculating xs
    scale = np.trapz(heights, centers) / np.trapz(true_y, true_x) # normalize distribution to histogram using numpy numerical integration
    
    ax[1].plot(display_x, display_y*scale, color = 'red', linewidth = 3)
    ax[1].set_xlabel(r'$cos(\theta_k)$')
    ax[1].set_ylabel(r'Candidates')
    
    phis = fit.bins[_bin]['phi']
    heights, edges, patches = ax[2].hist(phis, color = 'xkcd:powder blue')
    centers = 0.5*(edges[1]-edges[0]) + edges[:-1]
    display_x = np.linspace(edges.min(), edges.max(), 50) # xs to fill range of plot
    true_x = np.linspace(centers.min(), centers.max(), 50) # xs for calculating area
    display_y = fit.pdf_phi(display_x, s3 = s3, s9 = s9) # ys to match plotting xs
    true_y = fit.pdf_phi(true_x, s3 = s3, s9 = s9) # ys to match calculating xs
    scale = np.trapz(heights, centers) / np.trapz(true_y, true_x) # normalize distribution to histogram using numpy numerical integration
    
    ax[2].plot(display_x, display_y*scale, color = 'red', linewidth = 3)
    ax[2].set_xlabel(r'$\phi$')
    ax[2].set_ylabel(r'Candidates')
    
    
def plot_toy_data_observables():
    'finds the observables for each bin of the toy data set and plots across the bins'
    vals, errs = fit.toy_data_observables()
    fig, ax = plt.subplots(2, 4)
    ax[0][0].set_title(r'$A_{FB}$')
    ax[0][1].set_title(r'$F_L$')
    ax[0][2].set_title(r'$S_3$')
    ax[0][3].set_title(r'$S_4$')
    ax[1][0].set_title(r'$S_5$')
    ax[1][1].set_title(r'$S_7$')
    ax[1][2].set_title(r'$S_8$')
    ax[1][3].set_title(r'$S_9$')

    for i in range(9):
        if i == 2:
            ax[0][1].errorbar(range(len(fit.bins)), vals[:,i], xerr = np.array([.5 for n in range(len(fit.bins))]), yerr = errs[:,i], fmt = 'x', capsize = 4)
        elif i < 2:
            ax[0][i].errorbar(range(len(fit.bins)), vals[:,i], xerr = np.array([.5 for n in range(len(fit.bins))]), yerr = errs[:,i], fmt = 'x', capsize = 4)
            ax[0][i].set_xlabel('Bin number')
        elif i > 2:
            ax[int((i-1)/4)][(i-1)%4].errorbar(range(len(fit.bins)), vals[:,i], xerr = np.array([.5 for n in range(len(fit.bins))]), yerr = errs[:,i], fmt = 'x', capsize = 4)
            ax[int((i-1)/4)][(i-1)%4].set_xlabel('Bin number')