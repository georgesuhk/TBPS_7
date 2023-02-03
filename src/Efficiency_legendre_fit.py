#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import legendre
from tqdm import tqdm
from numba import jit
import time
from scipy import integrate

import warnings
warnings.filterwarnings("ignore")
#%%
#import data
data = pd.read_csv('acceptance_mc.csv')
data = data.dropna()
#%%
maxq2 = max(data['q2'])

def preprocess_jit(bin):
    #pre process data for numba
    bin['phi'] = bin['phi']/np.pi
    bin['q2'] = (bin['q2']- maxq2/2)/(maxq2/2)

    costhetal_ls = bin['costhetal'].tolist()
    costhetak_ls = bin['costhetak'].tolist()
    phi_ls = bin['phi'].tolist()
    q2_ls = bin['q2'].tolist()

    return [costhetal_ls, costhetak_ls, phi_ls, q2_ls]

@jit(nopython=True)
def manual_legendre(n, x):
    #need to manually define legendre polynomials as numba doesn't understand scipy
    #its also much faster to use explicit equation rather than general formula
    if n == 0:
        return 1
    elif n == 1:
        return x
    elif n == 2:
        return (1/2)*(3*x**2 - 1)
    elif n == 3:
        return (1/2)*(5*x**3 - 3*x)
    elif n == 4:
        return (1/8)*(35*x**4 - 30*x**2 +3)
    elif n == 5:
        return (1/8)*(63*x**5 - 70*x**3 + 15*x)
    elif n == 6:
        return (1/16)*(231*x**6 - 315*x**4 + 105*x**2 -5)
    elif n == 7:
        return (1/16)*(429*x**7 - 693*x**5 + 315*x**3 - 35*x)
    elif n == 8:
        return (1/128)*(6435*x**8 - 12012*x**6 + 6930*x**4 - 1260*x**2 + 35)

@jit(nopython=True)
def get_efficiency_coeff_kress(costhetal_ls, costhetak_ls, phi_ls, q2_ls, i_max=5, j_max=5, m_max=5, n_max=5, mode='total_unweighted'):
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
                            
        return c_coeff

@jit(nopython=True)
def get_efficiency_kress(costhetal, costhetak, phi, q2, coeff_ls, i_max=5, j_max=5, m_max=5, n_max=5, mode='total_unweighted'):
    i_max += 1
    j_max += 1
    m_max += 1
    n_max += 1
    
    phi = phi/np.pi
    q2 = (q2-maxq2/2)/(maxq2/2)
    
    c_ls_pos = 0
    total_efficiency = 0

    if mode == 'total_unweighted':
        for i in range(0, i_max):
            for j in range(0, j_max):
                for m in range(0, m_max):
                    for n in range(0, n_max):
                        user_input_legendre_terms = manual_legendre(i,costhetal)*manual_legendre(j,costhetak)*manual_legendre(m,phi)*manual_legendre(n,q2)
                        total_efficiency += coeff_ls[c_ls_pos]*user_input_legendre_terms
                        c_ls_pos += 1
        return total_efficiency

@jit(nopython=True)
def project_1d(var_ls, N, func, func_params, projected_variable='q2'):
    #due to the way numba works, we cannot use global variables inside functions.
    #so instead, we pass the arguments/parameters of the efficiency function as input here
    output_ls = []

    for var in var_ls:
        integral = 0

        for i in range(0, N):
            x = np.random.uniform(-1,1)
            y = np.random.uniform(-1,1)
            z = np.random.uniform(-1,1)
            if projected_variable == 'q2':
                print(integral)
                integral += func(x, y, z, var, *func_params)
            elif projected_variable == 'costhetal':
                integral += func(var, x, y, z, *func_params)
            elif projected_variable == 'costhetak':
                integral += func(x, var, y, z, *func_params)
            elif projected_variable == 'phi':
                integral += func(x, y, var, z, *func_params)
        
        output_ls.append(integral/N*2*2*2)
        print('done.')

    return output_ls
#%%
#Testing Kress' method
#Preprocess data for numba
preprocessed_data = preprocess_jit(data)
# %%
#Get coefficients.
start = time.time()
total_kress_coeff = get_efficiency_coeff_kress(*preprocessed_data, mode='total_unweighted', i_max=5, j_max=6, m_max=5, n_max=4)
end = time.time()
print("Elapsed = %s" % (end - start))

#%%
function_params = (total_kress_coeff, 5, 6, 5, 4, 'total_unweighted')
# q2_ls = np.linspace(-1,1,5)
# y_ls = project_1d(q2_ls, int(1e5), get_efficiency_kress, function_params, projected_variable='q2')
    

# %%
