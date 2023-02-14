#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit
from numba import njit
import time
import pickle 

import warnings
warnings.filterwarnings("ignore")
#%%
# import data
data = pd.read_csv('cleaned_acceptance.csv')
data = data.dropna()
# data = data[data['q2'] <= 18]
#%%
maxq2 = np.max(data['q2'])

def preprocess_jit(data):
    """Pre-processes the pandas data-frame by scaling the nessecary columns and converting
    them to lists.

    Args:
        data (pd dataframe): The acceptance file.

    Returns:
        list: list of lists.
    """
    bin = data
    #pre process data for numba
    phi_ls = pd.Series(bin['phi']/np.pi).tolist()
    q2_ls = pd.Series((bin['q2']- maxq2/2)/(maxq2/2)).tolist()

    costhetal_ls = bin['costhetal'].tolist()
    costhetak_ls = bin['costhetak'].tolist()

    return [costhetal_ls, costhetak_ls, phi_ls, q2_ls]

@jit(nopython=True)
def manual_legendre(n, x):
    #need to manually define legendre polynomials as numba doesn't understand scipy
    #its also much faster to use explicit equation rather than general formula
    if n == 0:
        return x/x
    elif n == 1:
        return x
    elif n == 2:
        return (1/2)*(3*x**2 - 1.)
    elif n == 3:
        return (1/2)*(5*x**3 - 3*x)
    elif n == 4:
        return (1/8)*(35*x**4 - 30*x**2 + 3.)
    elif n == 5:
        return (1/8)*(63*x**5 - 70*x**3 + 15*x)
    elif n == 6:
        return (1/16)*(231*x**6 - 315*x**4 + 105*x**2 - 5.)
    elif n == 7:
        return (1/16)*(429*x**7 - 693*x**5 + 315*x**3 - 35*x)
    elif n == 8:
        return (1/128)*(6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35.)

@jit(nopython=True)
def get_efficiency_coeff_kress(costhetal_ls, costhetak_ls, phi_ls, q2_ls, i_max=5, j_max=5, m_max=5, n_max=5, mode='total_unweighted'):
    """Gets the c_ijmn coefficients.

    Args:
        costhetal_ls (list): costhetal data
        costhetak_ls (list): costhetak data
        phi_ls (list): phi data
        q2_ls (list): q2 data
        i_max (int, optional): Max polynomial order for costhetal. Defaults to 5.
        j_max (int, optional): Max polynomial order for costhetak. Defaults to 5.
        m_max (int, optional): Max polynomial ordre for phi. Defaults to 5.
        n_max (int, optional): Max polynomial order fo q2.. Defaults to 5.
        mode (str, optional): Selects the whether to take into account q2 distribution. Defaults to 'total_unweighted'.

    Returns:
        list: List of c_ijmn coefficients.
    """
    i_max += 1
    j_max += 1
    m_max += 1
    n_max += 1

    c_coeff = []
    
    iterations = 0
    total_iterations = i_max*j_max*m_max*n_max

    #Total efficiency
    if mode == 'total_unweighted':
        for i in range(0, i_max):
            for j in range(0, j_max):
                for m in range(0, m_max):
                    for n in range(0, n_max):
                        c = 0
                        for e in range(0, len(costhetal_ls)):
                            c += manual_legendre(i, costhetal_ls[e])*manual_legendre(j, costhetak_ls[e])*manual_legendre(m, phi_ls[e])*manual_legendre(n, q2_ls[e])
                        c *= (2*i+1)*(2*j+1)*(2*m+1)*(2*n+1)/(2**4)
                        c /= len(costhetal_ls)
                        c_coeff.append(c)
                        iterations += 1
            # print(f'Obtained {iterations}/{total_iterations} Coefficients.')
                            
        return np.array(c_coeff)

#@jit(nopython=True)
def get_efficiency_kress(costhetal, costhetak, phi, q2, coeff_ls, i_max=5, j_max=5, m_max=5, n_max=5):
    """A function of four variables: costhetal, costhetak, phi, and q2, which outputs a single scalar for the effiency at that point.
    For this to work, coeff_ls must be generated using get_efficiency_coeff_kress, using the same values of i,j,m,n_max

    Args:
        costhetal (float): costhetal
        costhetak (float): costhetak
        phi (float): phi
        q2 (float): q2
        coeff_ls (list): List of coefficients generated using get_efficiency_coeff_kress with same ijmn_max values.
        i_max (int, optional): Must be same as the one used to generate coeff_ls. Defaults to 5.
        j_max (int, optional): Must be same as the one used to generate coeff_ls. Defaults to 5.
        m_max (int, optional): Must be same as the one used to generate coeff_ls. Defaults to 5.
        n_max (int, optional): Must be same as the one used to generate coeff_ls.. Defaults to 5.

    Returns:
        _type_: _description_
    """
    i_max += 1
    j_max += 1
    m_max += 1
    n_max += 1
    
    phi = phi/np.pi
    q2 = (q2-maxq2/2)/(maxq2/2)
    
    c_ls_pos = 0
    total_efficiency = np.zeros(np.shape(costhetal))

    
    for i in range(0, i_max):
        for j in range(0, j_max):
            for m in range(0, m_max):
                for n in range(0, n_max):
                    user_input_legendre_terms = manual_legendre(i,costhetal)*manual_legendre(j,costhetak)*manual_legendre(m,phi)*manual_legendre(n,q2)
                    total_efficiency += (coeff_ls[c_ls_pos]*user_input_legendre_terms)
                    c_ls_pos += 1
    return total_efficiency

@jit(nopython=True)
def effiency_costhetal(costhetal_ls, func, func_params, costhetak_lim, phi_lim, q2_lim, N_costhetak, N_phi, N_q2):
    hx = (costhetak_lim[1]-costhetak_lim[0])
    hx = hx/N_costhetak
    hy = (phi_lim[1]-phi_lim[0])
    hy = hy/N_phi
    hz = (q2_lim[1]-q2_lim[0])
    hz = hz/N_q2
    x_ls = np.linspace(costhetak_lim[0], costhetak_lim[1], N_costhetak)
    y_ls = np.linspace(phi_lim[0], phi_lim[1], N_phi)
    z_ls = np.linspace(q2_lim[0], q2_lim[1], N_q2)

    output = []
    for costhetal in costhetal_ls:
        integral = 0
        for i in range(1, N_costhetak-1):
            for j in range(1, N_phi-1):
                for k in range(1, N_q2-1):
                    integral += func(costhetal, x_ls[i], y_ls[j], z_ls[k], *func_params)
        integral = integral*hx*hy*hz
        output.append(integral)
    return output

@jit(nopython=True)
def effiency_costhetak(costhetak_ls, func, func_params, costhetal_lim, phi_lim, q2_lim, N_costhetal, N_phi, N_q2):
    hx = (costhetal_lim[1]-costhetal_lim[0])
    hx = hx/N_costhetal
    hy = (phi_lim[1]-phi_lim[0])
    hy = hy/N_phi
    hz = (q2_lim[1]-q2_lim[0])
    hz = hz/N_q2
    x_ls = np.linspace(costhetal_lim[0], costhetal_lim[1], N_costhetal)
    y_ls = np.linspace(phi_lim[0], phi_lim[1], N_phi)
    z_ls = np.linspace(q2_lim[0], q2_lim[1], N_q2)

    output = []
    for costhetak in costhetak_ls:
        integral = 0
        for i in range(1, N_costhetal-1):
            for j in range(1, N_phi-1):
                for k in range(1, N_q2-1):
                    integral += func(x_ls[i], costhetak, y_ls[j], z_ls[k], *func_params)
        integral = integral*hx*hy*hz
        output.append(integral)
    return output

@jit(nopython=True)
def effiency_phi(phi_ls, func, func_params, costhetal_lim, costhetak_lim, q2_lim, N_costhetal, N_costhetak, N_q2):
    hx = (costhetal_lim[1]-costhetal_lim[0])
    hx = hx/N_costhetal
    hy = (costhetak_lim[1]-costhetak_lim[0])
    hy = hy/N_costhetak
    hz = (q2_lim[1]-q2_lim[0])
    hz = hz/N_q2
    x_ls = np.linspace(costhetal_lim[0], costhetal_lim[1], N_costhetal)
    y_ls = np.linspace(costhetak_lim[0], costhetak_lim[1], N_costhetak)
    z_ls = np.linspace(q2_lim[0], q2_lim[1], N_q2)

    output = []
    for phi in phi_ls:
        integral = 0
        for i in range(1, N_costhetal-1):
            for j in range(1, N_costhetak-1):
                for k in range(1, N_q2-1):
                    integral += func(x_ls[i], y_ls[i], phi, z_ls[k], *func_params)
        integral = integral*hx*hy*hz
        output.append(integral)
    return output

def relative_histogram_generator(data, num_datapoints=100):
    """Returns the relative count and each bin location.
    Args:
        data (list): The data we want to make a histogram out of.
        num_datapoints (int, optional): Number of datapoints we want on histogram. Defaults to 100.
    Returns:
        list, list: Relative count, and bin location.
    """
    hist, bin_edges = np.histogram(data, bins=num_datapoints)
    hist = hist/max(hist) #normalizes it since we want to plot relative efficiency

    bin_locations = []
    for i in range(1, len(bin_edges)):
        bin_locations.append((bin_edges[i-1]+bin_edges[i])/2)
    
    return hist, bin_locations
#%%

vals = preprocess_jit(data)
coeffs = get_efficiency_coeff_kress(*vals)