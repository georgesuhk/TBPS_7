#%%
# coded by Henry and Varim

# import relavent modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import scipy
from tqdm import tqdm
import json
#%%
# import datatset
df_before = pd.read_csv('acceptance_mc.csv') # before selection
df_after = pd.read_csv('acceptance_after.csv') # after selection

#%%
#polynomials to fit
n6_polynomial = lambda x, a,b,c,d,e,f,g : a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x*2 + f*x + g
n6_polynomial_even = lambda x, a,b,c,d : a*x**6 + b*x**4 + c*x**2 + d

#added by henry
def q2_binning_sm(data, bin_ranges):
    """Returns a list of pandas data frames. Each data frame has q2 within the range specified by
    bin_ranges. For example, if bin_ranges = [0,1,2], a list of two data frames will be returned,
    the first element containing data with 0 <= q2 < 1, and the second element containing data with
    1 <= q2 < 2
    Args:
        data (pd dataframe): Acceptance.csv after applying pre-selection and ML.
        bin_ranges (dict): Dictionary of q2 bin ranges.
    Returns:
        list: A list of pd dataframes, each data frame having q2 in a certain range.
    """
    return [data[(data['q2'] >= bin_range[0]) & (data['q2'] <= bin_range[1])] for bin_range in bin_ranges.values()]
    
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
        bin_ranges (list): Dictionary of bin ranges.
        N (list): Number of datapoints to we want on each histogram.
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
        y_costhetak, x_costhetak = relative_histogram_generator(costhetak, num_datapoints=N[i])
        popt_costhetak, cov_costhetak = curve_fit(n6_polynomial, x_costhetak, y_costhetak)
        if plotdata == True:
            plt.plot(x_costhetak, y_costhetak, '.', c='black')
            x = np.linspace(-1, 1, 10000)
            plt.plot(x, n6_polynomial(x, *popt_costhetak), c='black')
            plt.xlabel(r'$cos(\theta_{k})$')
            plt.ylabel('Relative efficiency')
            plt.title(f'{bin_ranges[i][0]} < q^2 < {bin_ranges[i][1]}')
            plt.grid()
            plt.show()
        
        costhetal = bin['costhetal']
        y_costhetal, x_costhetal = relative_histogram_generator(costhetal, num_datapoints=N[i])
        popt_costhetal, cov_costhetal = curve_fit(n6_polynomial_even, x_costhetal, y_costhetal)
        if plotdata == True:
            plt.plot(x_costhetal, y_costhetal, '.', c='black')
            x = np.linspace(-1, 1, 10000)
            plt.plot(x, n6_polynomial_even(x, *popt_costhetal), c='black')
            plt.xlabel(r'$cos(\theta_{l})$')
            plt.ylabel('Relative efficiency')
            plt.title(f'{bin_ranges[i][0]} < q^2 < {bin_ranges[i][1]}')
            plt.grid()
            plt.show()

        phi = bin['phi']
        y_phi, x_phi = relative_histogram_generator(phi, num_datapoints=N[i])
        popt_phi, cov_phi = curve_fit(n6_polynomial_even, x_phi, y_phi)
        if plotdata == True:
            plt.plot(x_phi, y_phi, '.', c='black')
            x = np.linspace(-np.pi, np.pi, 10000)
            plt.plot(x, n6_polynomial_even(x, *popt_phi), c='black')
            plt.xlabel(r'$\phi$')
            plt.ylabel('Relative efficiency')
            plt.title(f'{bin_ranges[i][0]} < q^2 < {bin_ranges[i][1]}')
            plt.grid()
            plt.show()

        #Save the fit parameters into array
        costhetak_popt_ls.append(popt_costhetak.tolist())
        costhetal_popt_ls.append(popt_costhetal.tolist())
        phi_popt_ls.append(popt_phi.tolist())
        costhetak_cov_ls.append(cov_costhetak.tolist())
        costhetal_cov_ls.append(cov_costhetal.tolist())
        phi_cov_ls.append(cov_phi.tolist())

    if covariance == True:
        return costhetak_popt_ls, costhetal_popt_ls, phi_popt_ls, costhetak_cov_ls, costhetal_cov_ls, phi_cov_ls
    else:
        return costhetak_popt_ls, costhetal_popt_ls, phi_popt_ls
        
#define the bootstrap function added by henry
def get_bootstrap_params(df_after, num, bin_ranges, raw_popt_costhetak_ls, raw_popt_costhetal_ls, raw_popt_phi_ls):
    """This is the function that generates 500 pseudo-events from the raw dataset.
    Args:
        df_after(pandas dataframe): raw dataset
        num(int): number of pseudo-events we want
        bin_ranges(dict): the standard model bin ranges
    Returns:
        3 3D arrays which are a list of parameters for each q^2 bins for each pseudo_event list of 500 pseudo-events.
    """
    params_costhetak, params_costhetal, params_phi = [], [], [] 
    for i in tqdm(range(num)):
        event = df_after.sample(frac=1, replace=True, random_state=1)
        bins_ls = q2_binning_sm(event, bin_ranges)
        num_of_bins_ls = [int(round(len(bins)/1000, -1)) for bins in bins_ls] #this is here to set the amount of bins for each q^2 bins  
        popt_costhetak_ls, popt_costhetal_ls, popt_phi_ls = get_efficiency(bins_ls, bin_ranges, N=num_of_bins_ls, covariance=False, plotdata=False)
        params_costhetak.append(popt_costhetak_ls)
        params_costhetal.append(popt_costhetal_ls)
        params_phi.append(popt_phi_ls)
    params_costhetak.append(raw_popt_costhetak_ls)
    params_costhetal.append(raw_popt_costhetal_ls)
    params_phi.append(raw_popt_phi_ls)
    with open('acceptance_k_params.json', 'w') as outfile:
        json.dump(params_costhetak, outfile)
    with open('acceptance_l_params.json', 'w') as outfile:
        json.dump(params_costhetal, outfile)
    with open('acceptance_phi_params.json', 'w') as outfile:
        json.dump(params_phi, outfile)
    #return params_costhetak, params_costhetal, params_phi
    
def main():
    #SM predictions bin ranges added by henry
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
    #using q2_binning added by henry
    bins_ls = q2_binning_sm(df_before, bin_ranges)
    num_of_bins_ls = [int(round(len(bins)/1000, -1)) for bins in bins_ls]
    #print(num_of_bins_ls)
    popt_costhetak_ls, popt_costhetal_ls, popt_phi_ls = get_efficiency(bins_ls, bin_ranges, N=num_of_bins_ls, covariance=False, plotdata=False)
    get_bootstrap_params(df_after = df_after, num = 500, bin_ranges = bin_ranges, raw_popt_costhetak_ls = popt_costhetak_ls, raw_popt_costhetal_ls = popt_costhetal_ls, raw_popt_phi_ls = popt_phi_ls)
#%%
main()
# %%
