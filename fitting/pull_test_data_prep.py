# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:09:47 2023

@author: victo
"""

# pull test data for Fei

import numpy as np

import multiD_fitting as fit_mD
import fitting_functions as fit_proj

bins = fit_mD.bins
bin_ranges = fit_mD.bin_ranges

mvals = fit_mD.vals # observable values from multidimensional fitting
merrs = fit_mD.errs # observable errors from multidimensional fitting

pvals = fit_proj.vals # observable values from projected pdf fitting
perrs = fit_proj.errs # observable errors from projected pdf fitting


def get_pull_inputs(_bin):
    ctl = bins[_bin]['ctl']
    ctk = bins[_bin]['ctk']
    phi = bins[_bin]['phi']
    
    ctl_heights, ctl_edges = np.histogram(ctl, bins = 30)
    ctk_heights, ctk_edges = np.histogram(ctk, bins = 30)
    phi_heights, phi_edges = np.histogram(phi, bins = 30)
    
    afb_err = perrs[_bin][0]
    fl_ctl_err = perrs[_bin][1]
    ctl_centers = 0.5*(ctl_edges[1]-ctl_edges[0]) + ctl_edges[1:]
    ctl_fit_heights = fit_proj.pdf_ctl(ctl_centers, afb=pvals[_bin][0], fl=pvals[_bin][1])
    ctl_fit_errors = np.sqrt((9/256)*(1+ctl_centers**2)*fl_ctl_err**2 + ctl_centers**2*afb_err**2)
    ctl_results = np.array([ctl_centers, ctl_heights, ctl_fit_heights, ctl_fit_errors])
    
    fl_ctk_err = perrs[_bin][2]
    ctk_centers = 0.5*(ctk_edges[1]-ctl_edges[0]) + ctk_edges[1:]
    ctk_fit_heights = fit_proj.pdf_ctk(ctk_centers, fl=pvals[_bin][2])
    ctk_fit_errors = 0.75 * (3*ctk_centers**2 - 1) * fl_ctk_err
    ctk_results = np.array([ctk_centers, ctk_heights, ctk_fit_heights, ctk_fit_errors])
    
    s3_err = perrs[_bin][3]
    s9_err = perrs[_bin][8]
    phi_centers = 0.5*(phi_edges[1]-phi_edges[0]) + phi_edges[1:]
    phi_fit_heights = fit_proj.pdf_phi(phi_centers, s3=pvals[_bin][3], s9=pvals[_bin][8])
    phi_fit_errors = np.sqrt(((1/np.pi) * np.cos(2*phi_centers))**2 * s3_err ** 2 + ((1/np.pi) * np.sin(2*phi_centers)) ** 2 * s9_err ** 2)
    phi_results = np.array([phi_centers, phi_heights, phi_fit_heights, phi_fit_errors])
    
    return ctl_results, ctk_results, phi_results



