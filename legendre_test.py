#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Efficiency_legendre_fit import *
from scipy.integrate import nquad
from numba import jit
#%%
"""
Example code
"""
#The data has already been loaded when importing the module.

#Pre-process the data to be compatible with jit.
preprocessed_data = preprocess_jit(data) 

#Get efficiency coefficients.
total_kress_coeff = get_efficiency_coeff_kress(*preprocessed_data, mode='total_unweighted', i_max=5, j_max=6, m_max=5, n_max=4)
#%%
bin_ranges = {
        0: [0.1, 0.98],
        1: [1.1, 2.5],
        2: [2.5, 4.0],
        3: [4.0, 6.0],
        4: [6.0, 8.0],
        5: [15.0, 17.0],
        6: [17.0, 19.0],
        7: [11.0, 12.5],
        8: [1.0, 6.0],
        9: [15.0, 19.0]
    }

for key in bin_ranges:
    lowq = bin_ranges[key][0]
    highq = bin_ranges[key][1]

    #Project the 4d function onto costhetal, over the ranges [-1,1], [-np.pi, np.pi], [lowq,highq]
    costhetal_ls = np.linspace(-1, 1, 100)
    start = time.time()
    func_params = (total_kress_coeff, 5, 6, 5, 4)
    y = effiency_costhetal(costhetal_ls, get_efficiency_kress, func_params, [-1,1], [-np.pi, np.pi], [lowq,highq], 10, 10, 10)
    end = time.time()
    print("Elapsed = %s" % (end - start))

    #Plot the result overlaying data histogram
    copy = data
    copy = copy[copy['q2']>lowq]
    copy = copy[copy['q2']<highq]
    y_hist, x_hist = relative_histogram_generator(copy['costhetal'], 100)
    y = np.array(y)
    y = y/max(y)*max(y_hist)
    plt.plot(x_hist, y_hist, '.')
    plt.plot(costhetal_ls, y)
    plt.title(f'{lowq}<q2<{highq}')
    plt.xlabel(r'$cos(\theta_{l})$')
    plt.ylabel(r'Relative count')
    plt.show()
# %%
