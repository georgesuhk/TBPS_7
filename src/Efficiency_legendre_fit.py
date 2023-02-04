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
import pickle 

import warnings
warnings.filterwarnings("ignore")
#%%
#import data
data = pd.read_csv('acceptance_mc.csv')
data = data.dropna()
#%%
maxq2 = max(data['q2'])

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
                            
        return c_coeff

@jit(nopython=True)
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
    total_efficiency = 0

    
    for i in range(0, i_max):
        for j in range(0, j_max):
            for m in range(0, m_max):
                for n in range(0, n_max):
                    user_input_legendre_terms = manual_legendre(i,costhetal)*manual_legendre(j,costhetak)*manual_legendre(m,phi)*manual_legendre(n,q2)
                    total_efficiency += coeff_ls[c_ls_pos]*user_input_legendre_terms
                    c_ls_pos += 1
    return total_efficiency

@jit(nopython=True)
def project_1d(var_ls, N, func, func_params, projected_variable, costhetal_range, costhetak_range, phi_range, q2_range):
    """Projects the 4d efficiency function onto 1d. The dimension to project onto is specified by projected_variable. The range of the other three variables
    should also be specified.

    This method uses Monte Carlo integration, which always has error in the order of O(N^-0.5).

    Suggestion: This method is quite slow. For var_ls of length 50, it takes about 200s. We recommend having len(var_ls) between 50-100, then interpolating the
    output to obtain the efficiency at any arbitrary value on that dimension.

    Args:
        var_ls (list): A list of the independent variable we are projecting onto, e.g. q2_ls = np.linspace(0, 18, 100)
        N (int): Number of random samples for Monte Carlo integration.
        func (function): The get_efficiency_kress method.
        func_params (tuple): Parameters for the get_efficiency_kress method.
        projected_variable (str, optional): The dimension to project onto, e.g. 'costhetak'. Defaults to 'q2'.
        costhetal_range (list, optional): Integration range for costhetal. Defaults to [-1,1].
        costhetak_range (list, optional): Integration range for costhetak. Defaults to [-1,1].
        phi_range (list, optional): Integration range for phi. Defaults to [-1,1].
        q2_range (list, optional): Integration range for q2. Defaults to [-1,1].

    Returns:
        list: A list of effiency values at each point in var_ls. Note, these efficiency values are not scaled to be between 0 and 1. This can
        simply be done by converting the output to np.array and do output/max(output).
    """
    #due to the way numba works, we cannot use global variables inside functions.
    #so instead, we pass the arguments/parameters of the efficiency function as input here

    output_ls = []

    if projected_variable == 'q2':
        norm = 1/N*(costhetal_range[1]-costhetal_range[0])*(costhetak_range[1]-costhetak_range[0])*(phi_range[1]-phi_range[0])
        xmin, xmax = costhetal_range
        ymin, ymax = costhetak_range
        zmin, zmax = phi_range
        for var in var_ls:
            integral = 0
            for i in range(0, N):
                x = np.random.uniform(xmin, xmax)
                y = np.random.uniform(ymin, ymax)
                z = np.random.uniform(zmin, zmax)
                integral += func(x, y, z, var, *func_params)
            output_ls.append(integral/norm)
    elif projected_variable == 'costhetal':
        norm = 1/N*(q2_range[1]-q2_range[0])*(costhetak_range[1]-costhetak_range[0])*(phi_range[1]-phi_range[0])
        for var in var_ls:
            integral = 0
            for i in range(0, N):
                x = np.random.uniform(costhetak_range[0], costhetak_range[1])
                y = np.random.uniform(phi_range[0], phi_range[1])
                z = np.random.uniform(q2_range[0], q2_range[1])
                integral += func(var, x, y, z, *func_params)
            output_ls.append(integral/norm)
    elif projected_variable == 'costhetak':
        norm = 1/N*(q2_range[1]-q2_range[0])*(costhetal_range[1]-costhetal_range[0])*(phi_range[1]-phi_range[0])
        for var in var_ls:
            integral = 0
            for i in range(0, N):
                x = np.random.uniform(costhetal_range[0], costhetal_range[1])
                y = np.random.uniform(phi_range[0], phi_range[1])
                z = np.random.uniform(q2_range[0], q2_range[1])
                integral += func(x, var, y, z, *func_params)
            output_ls.append(integral/norm)
    elif projected_variable == 'phi':
        norm = 1/N*(q2_range[1]-q2_range[0])*(costhetal_range[1]-costhetal_range[0])*(costhetak_range[1]-costhetak_range[0])
        for var in var_ls:
            integral = 0
            for i in range(0, N):
                x = np.random.uniform(costhetal_range[0], costhetal_range[1])
                y = np.random.uniform(costhetak_range[0], costhetak_range[1])
                z = np.random.uniform(q2_range[0], q2_range[1])
                integral += func(x, y, var, z, *func_params)
            output_ls.append(integral/norm)

    return output_ls

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
"""
Example code
"""
#Pre-process the data. This only has to be done once.
preprocessed_data = preprocess_jit(data)
# %%
#Get coefficients c_ijmn. This only has to be done once.
start = time.time()
total_kress_coeff = get_efficiency_coeff_kress(*preprocessed_data, mode='total_unweighted', i_max=5, j_max=6, m_max=5, n_max=4)
end = time.time()
print("Elapsed = %s" % (end - start))
#%%
#Visualize the efficieny by projecting the 4d function onto 1d.
start = time.time()
function_params = (total_kress_coeff, 5, 6, 5, 4) #Specifiy the function parameters to pass into
q2_ls = np.linspace(0, max(data['q2']), 50) #Create a list for q2 we would like to project the efficiencies onto.
#Project 4d efficiencies onto the q2 values specified on the previouse line.
#Note: the q2 range is not actually used, we just specified [-1,1] as a placeholder.
q2_y_ls = project_1d(q2_ls, int(1e5), get_efficiency_kress, function_params, projected_variable='q2', costhetak_range=[-1,1], costhetal_range=[-1,1], phi_range=[-1,1], q2_range=[-1,1]) 
end = time.time()
print("Elapsed = %s" % (end - start)) 

# %%
#Plotting the results
#Projection onto q2
q2_y_ls = np.array(q2_y_ls) #Converts q2_y_ls to a np.array.
q2_y_ls /= max(q2_y_ls) #Scale it so that we get an efficiency in the range 0 to 1.

q2_y_data, q2_x_data = relative_histogram_generator(data['q2'], num_datapoints=75) #Generate relative histogram bin centers and counts of q2 from data.

#Plot the results.
plt.plot(q2_x_data, q2_y_data, '.', label='Data')
plt.plot(q2_ls, q2_y_ls, label='4d Eff. projected onto 1d')
plt.xlabel(r'$q^{2}$')
plt.ylabel('Relative frequency')
plt.legend()
plt.ylim([0, 1])
plt.show()
# %%
#We can do the same steps to project onto costhetal, costhetak, and phi.
#Note the integration limits of q2.
maxq2 = max(data['q2'])

start = time.time()
function_params = (total_kress_coeff, 5, 6, 5, 4) 
costhetal_ls = np.linspace(-1, 1, 50) 
costhetal_y_ls = project_1d(costhetal_ls, int(1e5), get_efficiency_kress, function_params, projected_variable='q2', costhetak_range=[-1,1], costhetal_range=[-1,1], phi_range=[-1,1], q2_range=[0.0, maxq2]) 
end = time.time()
print("Elapsed = %s" % (end - start)) 

start = time.time()
function_params = (total_kress_coeff, 5, 6, 5, 4) 
costhetak_ls = np.linspace(-1, 1, 50) 
costhetak_y_ls = project_1d(costhetak_ls, int(1e5), get_efficiency_kress, function_params, projected_variable='q2', costhetak_range=[-1,1], costhetal_range=[-1,1], phi_range=[-1,1], q2_range=[0.0, maxq2]) 
end = time.time()
print("Elapsed = %s" % (end - start)) 

start = time.time()
function_params = (total_kress_coeff, 5, 6, 5, 4) 
phi_ls = np.linspace(-np.pi, np.pi, 50) 
phi_y_ls = project_1d(phi_ls, int(1e5), get_efficiency_kress, function_params, projected_variable='q2', costhetak_range=[-1,1], costhetal_range=[-1,1], phi_range=[-1,1], q2_range=[0.0, maxq2]) 
end = time.time()
print("Elapsed = %s" % (end - start)) 
# %%
costhetal_y_ls = np.array(costhetal_y_ls) 
costhetal_y_ls /= max(costhetal_y_ls) 
costhetal_y_data, costhetal_x_data = relative_histogram_generator(data['costhetal'], num_datapoints=75)
plt.plot(costhetal_x_data, costhetal_y_data, '.', label='Data')
plt.plot(costhetal_ls, costhetal_y_ls, label='4d Eff. projected onto 1d')
plt.xlabel(r'$cos(\theta_{l})$')
plt.ylabel('Relative frequency')
plt.legend()
plt.show()

costhetak_y_ls = np.array(costhetak_y_ls) 
costhetak_y_ls /= max(costhetak_y_ls) 
costhetak_y_data, costhetak_x_data = relative_histogram_generator(data['costhetak'], num_datapoints=75)
plt.plot(costhetak_x_data, costhetak_y_data, '.', label='Data')
plt.plot(costhetak_ls, costhetak_y_ls, label='4d Eff. projected onto 1d')
plt.xlabel(r'$cos(\theta_{k})$')
plt.ylabel('Relative frequency')
plt.legend()
plt.show()

phi_y_ls = np.array(phi_y_ls) 
phi_y_ls /= max(phi_y_ls) 
phi_y_data, phi_x_data = relative_histogram_generator(data['phi'], num_datapoints=75)
plt.plot(phi_x_data, phi_y_data, '.', label='Data')
plt.plot(phi_ls, phi_y_ls, label='4d Eff. projected onto 1d')
plt.xlabel(r'$\phi$')
plt.ylabel('Relative frequency')
plt.legend()
plt.show()

#%%
#Make a dictionary to store this data so we don't ever have to calculate them again.
#We simply take this and interpolate it to get values at any arbitrary point.
eff_1d = {}
eff_1d['q2_x'] = q2_ls
eff_1d['q2_y'] = q2_y_ls
eff_1d['costhetal_x'] = costhetal_ls
eff_1d['costhetal_y'] = costhetal_y_ls
eff_1d['costhetak_x'] = costhetak_ls
eff_1d['costhetak_y'] = costhetak_y_ls
eff_1d['phi_x'] = phi_ls
eff_1d['phi_y'] = phi_y_ls

with open('eff_1d.pkl', 'wb') as f:
    pickle.dump(eff_1d, f)