#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import legendre
from tqdm import tqdm
#%%
#import data
data = pd.read_csv('acceptance_mc.csv')
#%%
#polynomials to fit
n6_polynomial = lambda x, a,b,c,d,e,f,g : a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x*2 + f*x + g
n6_polynomial_even = lambda x, a,b,c,d : a*x**6 + b*x**4 + c*x**2 + d

def q2_binning(data, bin_ranges):
    """Returns a list of pandas data frames. Each data frame has q2 within the range specified by
    bin_ranges. For example, if bin_ranges = [0,1,2], a list of two data frames will be returned,
    the first element containing data with 0 <= q2 < 1, and the second element containing data with
    1 <= q2 < 2

    Args:
        data (pd dataframe): Acceptance.csv after applying pre-selection and ML.
        bin_ranges (list): List of q2 bin ranges.

    Returns:
        list: A list of pd dataframes, each data frame having q2 in a certain range.
    """
    bins_ls = []
    for i in range(1, len(bin_ranges)):
        bin = data[data['q2'] >= bin_ranges[i-1]]
        bin = bin[bin['q2'] < bin_ranges[i]]
        bins_ls.append(bin)
    return bins_ls


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

def histogram_generator(data, num_datapoints=100):
    hist, bin_edges = np.histogram(data, bins=num_datapoints)

    bin_locations = []
    for i in range(1, len(bin_edges)):
        bin_locations.append((bin_edges[i-1]+bin_edges[i])/2)
    
    return hist, bin_locations

def get_efficiency_coeff_kress(bin, i_max=5, j_max=5, m_max=5, n_max=5, mode='total_unwieghted'):
    #Rescale phi to be between -1 and 1
    bin['phi'] = bin['phi']/np.pi
    bin['q2'] = bin['q2']/max(bin['q2'])

    c_coeff = []
    
    #Total efficiency
    if mode == 'total_unweighted':
        for i in range(0, i_max):
            for j in range(0, j_max):
                for m in range(0, m_max):
                    for n in range(0, n_max):
                        c = 0
                        for e in range(0, len(bin)):
                            legendre_terms = legendre(i)(bin['costhetal'][e])*legendre(j)(bin['costhetak'][e])*legendre(m)(bin['phi'][e])*legendre(i)(bin['q2'][e])
                            c += (2*i+1)*(2*j+1)*(2*m+1)*(2*n+1)/(2**4)*legendre_terms
                            c /= len(bin)
                            c_coeff.append(c)
                            
        return c_coeff
    else:
        if mode == 'costhetal':
            for i in range(0, i_max):
                c = 0
                for e in tqdm(range(0, len(bin))):
                    c += (2*i+1)/2*legendre(i)(bin['costhetal'][e])
                c_coeff.append(c)
            return c_coeff
        
        if mode == 'costhetak':
            for j in range(0, j_max):
                c = 0
                for e in range(0, len(bin)):
                    c += (2*i+1)/2*legendre(i)(bin['costhetak'][e])
                c_coeff.append(c)
            return c_coeff

        if mode == 'phi':
            for i in range(0, m_max):
                c = 0
                for e in range(0, len(bin)):
                    c += (2*i+1)/2*legendre(i)(bin['phi'][e])
                c_coeff.append(c)
            return c_coeff

        if mode == 'q2':
            for i in range(0, n_max):
                c = 0
                for e in range(0, len(bin)):
                    c += (2*i+1)/2*legendre(i)(bin['q2'][e])
                c_coeff.append(c)
            return c_coeff

def get_efficiency_kress(coeff_ls, costhetal=0, costhetak=0, phi=0, q2=0, i_max=5, j_max=5, m_max=5, n_max=5, mode='total_unwieghted'):
    phi = phi/np.pi
    #q2 = q2/maximum value of q2 in the dataset
    
    c_ls_pos = 0
    total_efficiency = 0

    if mode == 'total_unweighted':
        for i in range(0, i_max):
            for j in range(0, j_max):
                for m in range(0, m_max):
                    for n in range(0, n_max):
                        user_input_legendre_terms = legendre(i)(costhetal)*legendre(j)(costhetak)*legendre(m)(phi)*legendre(n)(q2)
                        total_efficiency += coeff_ls[c_ls_pos]*user_input_legendre_terms
                        c_ls_pos += 1
        return total_efficiency

    if mode == 'costhetal':
        for i in range(0, i_max):
            print(c_ls_pos)
            total_efficiency += coeff_ls[c_ls_pos]*legendre(i)(costhetal)
            c_ls_pos += 1
        return total_efficiency
    
    if mode == 'costhetak':
        for i in range(0, j_max):
            total_efficiency += coeff_ls[c_ls_pos]*legendre(i)(costhetak)
            c_ls_pos += 1
        return total_efficiency

    if mode == 'phi':
        for i in range(0, m_max):
            total_efficiency += coeff_ls[c_ls_pos]*legendre(i)(phi)
            c_ls_pos += 1
        return total_efficiency
    
    if mode == 'q2':
        for i in range(0, m_max):
            total_efficiency += coeff_ls[c_ls_pos]*legendre(i)(q2)
            c_ls_pos += 1
        return total_efficiency

def get_efficiency(bins_ls, bin_ranges, N, covariance=False, plotdata=True):
    """Takes in a list of pandas data frames, as returned by q2_binning function. Takes in bin ranges,
    and fits the appropriate polynomials to costhetak, costhetal, and phi histograms. Returns the popt
    array for each q2 range in a list.

    Args:
        bins_ls (list): List of pd dataframes. Each element in a range of q2.
        bin_ranges (list): List of bin ranges.
        N (int): Number of datapoints to we want on each histogram.
        covariance (bool, optional): If set to True, returns the covariance matrix of the fits.. Defaults to False.
        plotdata (bool, optional): If set to True, the data will be plotted. Defaults to True.

    Returns:
        list: 3 list of lists. Eg. [A1, B1, ...], [A2, B2, ...], [A3, B3, ...] where A1 is the popt array for costhetak
        fit, and so on. If covariance == True: then 6 list of arrays will be returned, where each array is the covariance matrix
        for the corresponding fit.

    """
    costhetak_popt_ls = []
    costhetal_popt_ls = []
    phi_popt_ls = []
    costhetak_cov_ls = []
    costhetal_cov_ls = []
    phi_cov_ls = []

    for i, bin in enumerate(bins_ls):
        
        costhetak = bin['costhetak']
        y_costhetak, x_costhetak = relative_histogram_generator(costhetak, num_datapoints=N)
        popt_costhetak, cov_costhetak = curve_fit(n6_polynomial, x_costhetak, y_costhetak)
        if plotdata == True:
            plt.plot(x_costhetak, y_costhetak, '.', c='black')
            x = np.linspace(-1, 1, 10000)
            plt.plot(x, n6_polynomial(x, *popt_costhetak), c='black')
            plt.xlabel(r'$cos(\theta_{k})$')
            plt.ylabel('Relative efficiency')
            plt.title(f'{bin_ranges[i]} < q^2 < {bin_ranges[i+1]}')
            plt.grid()
            plt.show()
        
        costhetal = bin['costhetal']
        y_costhetal, x_costhetal = relative_histogram_generator(costhetal, num_datapoints=N)
        popt_costhetal, cov_costhetal = curve_fit(n6_polynomial_even, x_costhetal, y_costhetal)
        if plotdata == True:
            plt.plot(x_costhetal, y_costhetal, '.', c='black')
            x = np.linspace(-1, 1, 10000)
            plt.plot(x, n6_polynomial_even(x, *popt_costhetal), c='black')
            plt.xlabel(r'$cos(\theta_{l})$')
            plt.ylabel('Relative efficiency')
            plt.title(f'{bin_ranges[i]} < q^2 < {bin_ranges[i+1]}')
            plt.grid()
            plt.show()

        phi = bin['phi']
        y_phi, x_phi = relative_histogram_generator(phi, num_datapoints=N)
        popt_phi, cov_phi = curve_fit(n6_polynomial_even, x_phi, y_phi)
        if plotdata == True:
            plt.plot(x_phi, y_phi, '.', c='black')
            x = np.linspace(-np.pi, np.pi, 10000)
            plt.plot(x, n6_polynomial_even(x, *popt_phi), c='black')
            plt.xlabel(r'$\phi$')
            plt.ylabel('Relative efficiency')
            plt.title(f'{bin_ranges[i]} < q^2 < {bin_ranges[i+1]}')
            plt.grid()
            plt.show()

        #Save the fit parameters into array
        costhetak_popt_ls.append(popt_costhetak)
        costhetal_popt_ls.append(popt_costhetal)
        phi_popt_ls.append(popt_phi)
        costhetak_cov_ls.append(cov_costhetak)
        costhetal_cov_ls.append(cov_costhetal)
        phi_cov_ls.append(cov_phi)

    if covariance == True:
        return costhetak_popt_ls, costhetal_popt_ls, phi_popt_ls, costhetak_cov_ls, costhetal_cov_ls, phi_cov_ls
    else:
        return costhetak_popt_ls, costhetal_popt_ls, phi_popt_ls

def total_efficiency(costhetak, costhetal, phi, q2, q2_ranges, costhetak_popt_ls, costhetal_popt_ls, phi_popt_ls):
    """This is the final function for the total efficiency. It is a function of costhetak, costhetal, phi, and q2.
    We must supply the popt lists as generated using get_efficiency.

    Args:
        costhetak (float): Value of costhetak.
        costhetal (float): Value of costhetal.
        phi (float): Value of phi
        q2 (float): Value of q2
        q2_ranges (list): A list of the q2 ranges we used to find the popt lists.  
        costhetak_popt_ls (list): List of popt lists for costhetak fit, as generated using get_efficiency.
        costhetal_popt_ls (list): List of popt lists for costhetal fit, as generated using get_efficiency.
        phi_popt_ls (list): List of popt lists for phi fit, as generated using get_efficiency.

    Returns:
        float: The effieciency at that point.
    """
    #Check the value of q2 so that we can use the correct fitting
    if q2 < q2_ranges[0]:
        print(f'Value of q2={q2} is below the fitted range. Error.')
        return None
    if q2 > q2_ranges[-1]:
        print(f'Value of q^2={q2} is above the fitted range. Error.')
        return None

    for i, val in enumerate(q2_ranges):
        if q2 >= val:
            popt_costhetak = costhetak_popt_ls[i]
            popt_costhetal = costhetal_popt_ls[i]
            popt_phi = phi_popt_ls[i]
            return n6_polynomial(costhetak, *popt_costhetak)*n6_polynomial_even(costhetal, *popt_costhetal)*n6_polynomial_even(phi, *popt_phi)

#%%
data = data.dropna()

#bin_ranges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0,
                # 1.2,
                # 1.4,
                # 1.6,
                # 1.8,
                # 2.0,
                # 2.2,
                # 2.4,
                # 2.6,
                # 2.8000000000000003,
                # 3.0000000000000004,
                # 3.2000000000000006,
                # 3.400000000000001,
                # 3.600000000000001,
                # 3.800000000000001,
                # 4.000000000000001,
                # 4.200000000000001,
                # 4.400000000000001,
                # 4.600000000000001,
                # 4.800000000000002,
                # 5.000000000000002,
                # 5.200000000000002,
                # 5.400000000000002,
                # 5.600000000000002,
                # 5.8000000000000025,
                # 6.3, 6.8, 7.3, 7.8, 8.3, 8.8, 9.3, 9.8, 10.3, 10.8, 11.3, 11.8, 12.3, 12.8, 13.3, 13.8, 14.3, 14.8, 15.3, 15.8, 16.3, 16.8, 17.3, 17.8, 18.3, 18.8, 19.3, 19.8, 20.3, 20.8]

bin_ranges = bin_range = [0.1, 0.98, 2.5, 4, 6, 8, 15, 17, 19]

#Bin the data according to q2
bins_ls = q2_binning(data, bin_ranges)
#Compute popt lists
popt_costhetak_ls, popt_costhetal_ls, popt_phi_ls = get_efficiency(bins_ls, bin_ranges, N=100, covariance=False, plotdata=True)
#%%
#Run this function to get the efficiency value for any 
#%%
#Testing Kress' method
#First get the coefficients
costhetal_kress_coeff = get_efficiency_coeff_kress(data, mode='costhetal')
#%%
costhetal = np.linspace(-1,1, 1000)
y = get_efficiency_kress(costhetal_kress_coeff, costhetal=costhetal, mode='costhetal')
y = np.array(y)
y /= np.max(y)
plt.plot(costhetal, y)
y_hist, x_hist = relative_histogram_generator(data['costhetal'], 30)
plt.plot(x_hist, y_hist, '.')
plt.xlabel(r'$cos(\theta_{l})$')
plt.ylabel(r'$Relative frequency$')
plt.title('for all q2')
plt.show()


# %%
