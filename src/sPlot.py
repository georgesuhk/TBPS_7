# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:29:02 2023

@author: User
"""
import numpy as np
import zfit
from matplotlib import pyplot as plt
import pandas
from hepstats.splot import compute_sweights
#%%
tot = pandas.read_csv('total_dataset.csv')
#%%
b0m_tot = tot['B0_M']
b0p_tot = tot['B0_P']
#%%
plt.hist(b0m_tot, bins=100)
plt.show()
plt.hist(b0p_tot,bins=100)
plt.show()
#%%
mu = zfit.Parameter("mu", 5300,5200,5400)
sigma = zfit.Parameter("sigma",20,1,200)
lambd = zfit.Parameter("lambda", -0.002, -0.01, 0.0001)
obs = zfit.Space('mass', (5100, 5700))
signal_pdf = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)
comb_bkg_pdf = zfit.pdf.Exponential(lambd, obs=obs)
sig_yield = zfit.Parameter('sig_yield', 25000, 0, 3e5,step_size=1)  # step size: default is small, use appropriate
bkg_yield = zfit.Parameter('bkg_yield', 100000, 0, 3e5, step_size=1)
# Create the extended models
extended_sig = signal_pdf.create_extended(sig_yield)
extended_bkg = comb_bkg_pdf.create_extended(bkg_yield)
# The final model is the combination of the signal and background PDF
model = zfit.pdf.SumPDF([extended_bkg, extended_sig])
#%%
# Builds the loss.
data_sw = zfit.Data.from_numpy(obs=obs, array=np.array(b0m_tot))
nll_sw = zfit.loss.ExtendedUnbinnedNLL(model, data_sw)
# This parameter was useful in the simultaneous fit but not anymore so we fix it.
sigma.floating = False
# Minimizes the loss.
minimizer = zfit.minimize.Minuit(use_minuit_grad=True)
result_sw = minimizer.minimize(nll_sw)
print(result_sw.params)
#%%
weights = compute_sweights(model, data_sw)
print(weights)
#%%  
plt.hist(np.array(b0p_tot), bins=100, weights=weights[sig_yield],label='background removed') #shows the b0 momentum after the background has been removed
plt.hist(np.array(b0p_tot),bins=100,histtype="step",label='original')
plt.legend()
plt.title('histogram of B0 momentum')