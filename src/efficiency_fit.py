#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
#%%
#import data
data = pd.read_csv('acceptance_mc.csv')
#%%
#polynomials to fit (will revisit to change to Legendre Polynomial)
n6_polynomial = lambda x, a,b,c,d,e,f,g : a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x*2 + f*x + g
n6_polynomial_even = lambda x, a,b,c,d : a*x**6 + b*x**4 + c*x**2 + d

#added by henry
def q2_binning_sm(data, bin_ranges):
    return [data[(data['q2'] >= bin_range[0]) & (data['q2'] <= bin_range[1])] for bin_range in bin_ranges.values()]

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

#Bin the data according to q2
#bins_ls = q2_binning(data, bin_ranges)
#using q2_binning added by henry
bins_ls = q2_binning_sm(df_before, bin_ranges)

#Compute popt lists
popt_costhetak_ls, popt_costhetal_ls, popt_phi_ls = get_efficiency(bins_ls, bin_ranges, N=100, covariance=False, plotdata=False)
#%%
#Run this function to get the efficiency value for any 
