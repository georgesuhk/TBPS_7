# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:50:43 2023

@author: victo
"""

import numpy as np
import matplotlib.pyplot as plt
import fitting_functions as fit
import multiD_fitting as mult
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import real_mass_dist_fitting as mass
from tqdm import tqdm
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
    x_length = 10 # side length of the array
    y_length = 10
    xs = np.linspace(x_range[0], x_range[1], x_length)
    ys = np.linspace(y_range[0], y_range[1], y_length)
    x_coord_array = np.broadcast_to(xs, [y_length, x_length]) # a 2D array where each entry describes its own x coordinate
    y_coord_array = np.broadcast_to(ys, [y_length, x_length]).T # a 2D array where each entry describes its own y coordinate
    
    heightmap = np.zeros([y_length, x_length])
    
    for y_index, y_val in enumerate(tqdm(ys)):
        for x_index, x_val in enumerate(xs):
            heightmap[y_index][x_index] = func(x_val, y_val)
        
    contours = plt.contour(x_coord_array, y_coord_array, heightmap, levels=np.linspace(1899, 1901, 8))
    for i, label in enumerate(contours.cvalues):
        contours.collections[i].set_label(label)
    plt.legend(title = r'$log(\mathcal{L})$ contour values')

def proof_contour_plots(vals, errs, _bin):
    plt.subplot(221)
    plt.xlabel(r'$A_{FB}$')
    plt.ylabel(r'$F_{L}$')
    ranges = []
    for i, val in enumerate(vals[_bin]):
        ranges.append([val - 2*errs[_bin][i], val + 2*errs[_bin][i]])
    plt.plot(vals[_bin][0], vals[_bin][1], 'P', color = 'xkcd:coral', markersize = 10)
    contour_plot(ranges[0], ranges[1], lambda afb, fl:mult.log_likelihood(_bin, mass.find_fsig(_bin), afb, fl, *vals[_bin][2:]))
    plt.subplot(222)
    plt.xlabel(r'$S_3$')
    plt.ylabel(r'$S_4$')
    plt.plot(vals[_bin][2], vals[_bin][3], 'P', color = 'xkcd:coral', markersize = 10)
    contour_plot(ranges[2], ranges[3], lambda s3, s4:mult.log_likelihood(_bin, mass.find_fsig(_bin), vals[_bin][0], vals[_bin][1], s3, s4, *vals[_bin][4:]))
    plt.subplot(223)
    plt.xlabel(r'$S_5$')
    plt.ylabel(r'$S_7$')
    plt.plot(vals[_bin][4], vals[_bin][5], 'P', color = 'xkcd:coral', markersize = 10)
    contour_plot(ranges[4], ranges[5], lambda s5, s7:mult.log_likelihood(_bin, mass.find_fsig(_bin), *vals[_bin][:4], s5, s7, *vals[_bin][6:]))
    plt.subplot(224)
    plt.xlabel(r'$S_8$')
    plt.ylabel(r'$S_9$')
    plt.plot(vals[_bin][6], vals[_bin][7], 'P', color = 'xkcd:coral', markersize = 10)
    contour_plot(ranges[6], ranges[7], lambda s8, s9:mult.log_likelihood(_bin, mass.find_fsig(_bin), *vals[_bin][:6], s8, s9))
    
    
    
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

    
def plot_predictions(path, bin_ranges, ax, obs_type = 'S'):
    if obs_type == 'S':
        SM_vals = np.loadtxt(f'{path}/expected_observable_values.csv', delimiter = ',', skiprows = 1)[:,1:]
        SM_errs = np.loadtxt(f'{path}/expected_observable_errors.csv', delimiter = ',', skiprows = 1)[:,1:]
    elif obs_type == 'P':
        SM_vals = np.loadtxt(f'{path}/p_pred_val.csv', delimiter = ',', skiprows = 1)[:,1:]
        SM_errs = np.loadtxt(f'{path}/p_pred_err.csv', delimiter = ',', skiprows = 1)[:,1:]
    for i in range(8):
        vals = SM_vals[:,i]
        errs = SM_errs[:,i]
        rectangles = [Rectangle((bin_range[0], vals[n] - errs[n]), bin_range[1] - bin_range[0], 2*errs[n]) for n, bin_range in enumerate(bin_ranges)]
        pc = PatchCollection(rectangles, facecolor = 'xkcd:powder blue')
        ax[int(i/4)][int(i%4)].add_collection(pc)
        
        
def plot_real_observables(vals, stat_errs, bin_ranges):
    #add sys_errs to parameters for the double errorbar plot
    fig, ax = plt.subplots(2, 4)
    ax[0][0].set_title(r'$A_{FB}$')
    ax[0][1].set_title(r'$F_L$')
    ax[0][2].set_title(r'$S_3$')
    ax[0][3].set_title(r'$S_4$')
    ax[1][0].set_title(r'$S_5$')
    ax[1][1].set_title(r'$S_7$')
    ax[1][2].set_title(r'$S_8$')
    ax[1][3].set_title(r'$S_9$')
    bin_centers = np.array([(bin_range[1] + bin_range[0])/2 for bin_range in bin_ranges])
    bin_widths = np.array([(bin_range[1] - bin_range[0])/2 for bin_range in bin_ranges])
    plot_predictions('C:/Users/victo/TBPS_7/Data', bin_ranges, ax)
    for i in range(8):
        ax[int(i/4)][int(i%4)].errorbar(bin_centers, vals[:,i], xerr = bin_widths, yerr = stat_errs[:,i], fmt = 'o', capsize = 4, capthick=2, color = 'xkcd:coral', ecolor = 'xkcd:cobalt', elinewidth=2)
        #ax[int(i/4)][int(i%4)].errorbar(bin_centers, vals[:,i], fmt = 'o', yerr = sys_errs[:,i], capsize = 4, capthick=2, color = 'xkcd:coral', ecolor = 'xkcd:pinkish', elinewidth =2)
        ax[int(i/4)][int(i%4)].set_xlabel(r'$q^2$')
        ax[int(i/4)][int(i%4)].grid()

def plot_opti_observables(vals, errs, bin_ranges):
    fig, ax = plt.subplots(2, 4)
    ax[0][0].set_title(r'$F_L$')
    ax[0][1].set_title(r'$P_1$')
    ax[0][2].set_title(r'$P_2$')
    ax[0][3].set_title(r'$P_3$')
    ax[1][0].set_title(r'$P_4^\prime$')
    ax[1][1].set_title(r'$P_5^\prime$')
    ax[1][2].set_title(r'$P_6^\prime$')
    ax[1][3].set_title(r'$S_8^\prime$')
    plot_predictions('C:/Users/victo/TBPS_7/Data', bin_ranges, ax, 'P')
    bin_centers = np.array([(bin_range[1] + bin_range[0])/2 for bin_range in bin_ranges])
    bin_widths = np.array([(bin_range[1] - bin_range[0])/2 for bin_range in bin_ranges])
    for i in range(8):
        ax[int(i/4)][int(i%4)].errorbar(bin_centers, vals[:,i], xerr = bin_widths, yerr = errs[:,i], fmt = 'o', capsize = 4, capthick=2, color = 'xkcd:coral', ecolor = 'xkcd:cobalt', elinewidth=2)
        ax[int(i/4)][int(i%4)].set_xlabel(r'$q^2$')
        ax[int(i/4)][int(i%4)].grid()

def plot_asymmetries(vals, errs, bin_ranges):
    ax = [
        plt.subplot(241),
        plt.subplot(242),
        plt.subplot(243),
        plt.subplot(244),
        plt.subplot(234),
        plt.subplot(235),
        plt.subplot(236)
        ]
    ax[0].set_title(r'$A_3$')
    ax[1].set_title(r'$A_4$')
    ax[2].set_title(r'$A_5$')
    ax[3].set_title(r'$A_6$')
    ax[4].set_title(r'$A_7$')
    ax[5].set_title(r'$A_8$')
    ax[6].set_title(r'$A_9$')
    bin_centers = np.array([(bin_range[1] + bin_range[0])/2 for bin_range in bin_ranges])
    bin_widths = np.array([(bin_range[1] - bin_range[0])/2 for bin_range in bin_ranges])
    for i in range(7):
        ax[i].grid()
        ax[i].errorbar(bin_centers, vals[:,i], xerr = bin_widths, yerr = errs[:,i], fmt = 'o', capsize = 4, capthick=2, color = 'xkcd:coral', ecolor = 'xkcd:cobalt', elinewidth=2)
        
    
def plot_real_observables_proj(bin_ranges):
    vals, errs = fit.real_data_observables()
    fig, ax = plt.subplots(2, 4)
    ax[0][0].set_title(r'$A_{FB}$')
    ax[0][1].set_title(r'$F_L$')
    ax[0][2].set_title(r'$S_3$')
    ax[0][3].set_title(r'$S_4$')
    ax[1][0].set_title(r'$S_5$')
    ax[1][1].set_title(r'$S_7$')
    ax[1][2].set_title(r'$S_8$')
    ax[1][3].set_title(r'$S_9$')
    bin_centers = np.array([(bin_range[1] + bin_range[0])/2 for bin_range in bin_ranges])
    bin_widths = np.array([(bin_range[1] - bin_range[0])/2 for bin_range in bin_ranges])
    plot_predictions('C:/Users/victo/TBPS_7/Data/', bin_ranges, ax, 'S')
    for i in range(9):
        if i == 2:
            ax[0][1].errorbar(bin_centers, vals[:,i], xerr = bin_widths, yerr = errs[:,i], fmt = 'o', capsize = 4, capthick=2, color = 'xkcd:coral', ecolor = 'xkcd:burnt sienna', elinewidth=2)
        elif i < 2:
            ax[0][i].errorbar(bin_centers, vals[:,i], xerr = bin_widths, yerr = errs[:,i], fmt = 'o', capsize = 4, capthick=2, color = 'xkcd:coral', ecolor = 'xkcd:cobalt', elinewidth=2)
            ax[0][i].set_xlabel('Bin number')
        elif i > 2:
            ax[int((i-1)/4)][(i-1)%4].errorbar(bin_centers, vals[:,i], xerr = bin_widths, yerr = errs[:,i], fmt = 'o', capsize = 4, capthick=2, color = 'xkcd:coral', ecolor = 'xkcd:cobalt', elinewidth=2)
            ax[int((i-1)/4)][(i-1)%4].set_xlabel(r'$q^2$')
            
def plot_multiD_fit(vals, errs, axis):
    fig, ax = plt.subplots(2, 5)
    for i, bin_range in enumerate(mult.bin_ranges):  
        subax = ax[int(i/5)][int(i%5)]
        subax.set_title(r'$%.1f < q^2 < %.1f$' %(bin_range[0], bin_range[1]))
        sig_f = mass.find_fsig(i)
        bkg_f = 1 - sig_f
        if axis == 'ctl':
            bin_ctl = mult.bins[i]['ctl']
            heights, edges, patches = subax.hist(bin_ctl, label = 'Data', color = 'xkcd:powder blue')
            width = edges[1]-edges[0]
            centers = 0.5*width + edges[:-1]
            x = np.linspace(np.min(edges), np.max(edges), 200)
            f = fit.pdf_ctl(x, afb = vals[i][0], fl = vals[i][1], _bin=i)
            bkg = mass.bkg_dist(x, 0, 0, *mass.param_bins[i])
            subax.set_xlabel(r'$cos(\theta_l)$')
        elif axis == 'ctk':
            bin_ctk = mult.bins[i]['ctk']
            heights, edges, patches = subax.hist(bin_ctk, label = 'Data', color = 'xkcd:powder blue')
            width = edges[1]-edges[0]
            centers = 0.5*width + edges[:-1]
            x = np.linspace(np.min(edges), np.max(edges), 200)
            f = fit.pdf_ctk(x, fl = vals[i][1])
            bkg = mass.bkg_dist(0, x, 0, *mass.param_bins[i])
            subax.set_xlabel(r'$cos(\theta_k)$')
        elif axis == 'phi':
            bin_phi = mult.bins[i]['phi'] 
            heights, edges, patches = subax.hist(bin_phi, label = 'Data', color = 'xkcd:powder blue')
            width = edges[1]-edges[0]
            centers = 0.5*width + edges[:-1]
            x = np.linspace(np.min(edges), np.max(edges), 200)
            f = fit.pdf_phi(x, s3 = vals[i][2], s9=vals[i][7])
            bkg = mass.bkg_dist(0, 0, x, *mass.param_bins[i])
            
            subax.set_xlabel(r'$\phi$')
            
        bkg_norm = bkg_f / np.trapz(bkg, x)
        bkg *= bkg_norm
        f *= sig_f
        N = np.sum(width*heights)/np.trapz(f + bkg, x)
        s = (f + bkg)*N
        bkg *= N
        subax.plot(x, s, color = 'black', label = r'Projection of'+'\n'+r'fitted $d\Gamma/d\vec{\Omega}$', linewidth = 3)
        subax.plot(x, bkg, color = 'xkcd:coral', label = 'Fitted back-\nground', linewidth = 3)
        subax.set_ylabel('Count')
    ax[1][0].legend(labels= [r'Projection of'+'\n'+r'fitted $d^3\Gamma/d\vec{\Omega}$', 'Fitted back-\nground', 'Data'])

def plot_multiD_fit_1bin(vals, errs, _bin):
    fig, ax = plt.subplots(1, 3)
    sig_f = mass.find_fsig(_bin)
    bkg_f = 1 - sig_f
    
    #cos(theta_l)
    bin_ctl = mult.bins[_bin]['ctl']
    heights_ctl, edges_ctl, patches_ctl = ax[0].hist(bin_ctl, label = 'Data', color = 'xkcd:powder blue')
    width_ctl = edges_ctl[1]-edges_ctl[0]
    x_ctl = np.linspace(np.min(edges_ctl), np.max(edges_ctl), 200)
    f_ctl = fit.pdf_ctl(x_ctl, afb = vals[_bin][0], fl = vals[_bin][1], _bin=_bin)
    bkg_ctl = mass.bkg_dist(x_ctl, 0, 0, *mass.param_bins[_bin])
    bkg_norm_ctl = bkg_f / np.trapz(bkg_ctl, x_ctl)
    bkg_ctl *= bkg_norm_ctl
    f_ctl *= sig_f
    N_ctl = np.sum(width_ctl*heights_ctl)/np.trapz(f_ctl + bkg_ctl, x_ctl)
    s_ctl = (f_ctl + bkg_ctl)*N_ctl
    bkg_ctl *= N_ctl
    
    #cos(theta_k)
    bin_ctk = mult.bins[_bin]['ctk']
    heights_ctk, edges_ctk, patches_ctk = ax[1].hist(bin_ctk, label = 'Data', color = 'xkcd:powder blue')
    width_ctk = edges_ctk[1]-edges_ctk[0]
    x_ctk = np.linspace(np.min(edges_ctk), np.max(edges_ctk), 200)
    f_ctk = fit.pdf_ctk(x_ctk, fl = vals[_bin][1])
    bkg_ctk = mass.bkg_dist(0, x_ctk, 0, *mass.param_bins[_bin])
    bkg_norm_ctk = bkg_f / np.trapz(bkg_ctk, x_ctk)
    bkg_ctk *= bkg_norm_ctk
    f_ctk *= sig_f
    N_ctk = np.sum(width_ctk*heights_ctk)/np.trapz(f_ctk + bkg_ctk, x_ctk)
    s_ctk = (f_ctk + bkg_ctk)*N_ctk
    bkg_ctk *= N_ctk
    
    #phi
    bin_phi = mult.bins[_bin]['phi']
    heights_phi, edges_phi, patches_phi = ax[2].hist(bin_phi, label = 'Data', color = 'xkcd:powder blue')
    width_phi = edges_phi[1]-edges_phi[0]
    x_phi = np.linspace(np.min(edges_phi), np.max(edges_phi), 200)
    f_phi = fit.pdf_phi(x_phi, s3 = vals[_bin][2], s9=vals[_bin][7])
    bkg_phi = mass.bkg_dist(0, 0, x_phi, *mass.param_bins[_bin])
    bkg_norm_phi = bkg_f / np.trapz(bkg_phi, x_phi)
    bkg_phi *= bkg_norm_phi
    f_phi *= sig_f
    N_phi = np.sum(width_phi*heights_phi)/np.trapz(f_phi + bkg_phi, x_phi)
    s_phi = (f_phi + bkg_phi)*N_phi
    bkg_phi *= N_phi
    
    ax[0].plot(x_ctl, s_ctl, color = 'black', label = r'Projection of'+'\n'+r'fitted $d^3\Gamma/d\vec{\Omega}$', linewidth = 3)
    ax[0].plot(x_ctl, bkg_ctl, color = 'xkcd:coral', label = 'Fitted back-\nground', linewidth = 3)
    ax[0].set_ylabel('Count')
    ax[0].set_xlabel(r'$cos(\theta_l)$')
    #ax[0].legend()
    
    ax[1].plot(x_ctk, s_ctk, color = 'black', label = r'Projection of'+'\n'+r'fitted $d^3\Gamma/d\vec{\Omega}$', linewidth = 3)
    ax[1].plot(x_ctk, bkg_ctk, color = 'xkcd:coral', label = 'Fitted back-\nground', linewidth = 3)
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel(r'$cos(\theta_k)$')
    #ax[1].legend()
    
    ax[2].plot(x_phi, s_phi, color = 'black', label = r'Projection of'+'\n'+r'fitted $d^3\Gamma/d\vec{\Omega}$', linewidth = 3)
    ax[2].plot(x_phi, bkg_phi, color = 'xkcd:coral', label = 'Fitted back-\nground', linewidth = 3)
    ax[2].set_ylabel('Count')
    ax[2].set_xlabel(r'$\phi$')
    ax[2].legend()
    
    fig.suptitle(r'$%.1f < q^2 < %.1f$' %(mult.bin_ranges[_bin][0], mult.bin_ranges[_bin][1]))