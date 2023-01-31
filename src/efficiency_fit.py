#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
#%%
#import data

#q^2 binning code from eva and nikkhil

data = pd.read_csv('acceptance_after.csv')
#%%
#right now, using a random q^2 range to test my function
bin1 = data[data['q2'] > 7]
bin1 = bin1[bin1['q2'] < 12]
binned_data = [bin1]

#polynomials to fit
n6_polynomial = lambda x, a,b,c,d,e,f,g : a*x**6 + b*x**5 + c*x**4 + d*x**3 + e*x*2 + f*x + g
n6_polynomial_even = lambda x, a,b,c,d : a*x**6 + b*x**4 + c*x**2 + d

def total_efficiency(costhetak, costhetal, phi, popt_costhetak, popt_costhetal, popt_phi):
    return n6_polynomial(costhetak, *popt_costhetak)*n6_polynomial_even(costhetal, *popt_costhetal)*n6_polynomial_even(phi *popt_phi)

def relative_histogram_generator(data, num_datapoints=100):
    hist, bin_edges = np.histogram(data, bins=num_datapoints)
    hist = hist/max(hist) #normalizes it since we want to plot relative efficiency

    bin_locations = []
    for i in range(1, len(bin_edges)):
        bin_locations.append((bin_edges[i-1]+bin_edges[i])/2)
    
    return hist, bin_locations

costhetak_popt_ls = []
costhetal_popt_ls = []
phi_popt_ls = []
costhetak_cov_ls = []
costhetal_cov_ls = []
phi_cov_ls = []

for bin in binned_data:
    costhetak = bin['costhetak']
    y_costhetak, x_costhetak = relative_histogram_generator(costhetak)
    popt_costhetak, cov_costhetak = curve_fit(n6_polynomial, x_costhetak, y_costhetak)
    plt.plot(x_costhetak, y_costhetak, '.', c='black')
    x = np.linspace(-1, 1, 10000)
    plt.plot(x, n6_polynomial(x, *popt_costhetak), c='black')
    plt.xlabel(r'$cos(\theta_{k})$')
    plt.ylabel('Relative efficiency')
    plt.title('8 < q^2 < 12')
    plt.grid()
    plt.show()
    
    costhetal = bin['costhetal']
    y_costhetal, x_costhetal = relative_histogram_generator(costhetal)
    popt_costhetal, cov_costhetal = curve_fit(n6_polynomial_even, x_costhetal, y_costhetal)
    plt.plot(x_costhetal, y_costhetal, '.', c='black')
    x = np.linspace(-1, 1, 10000)
    plt.plot(x, n6_polynomial_even(x, *popt_costhetal), c='black')
    plt.xlabel(r'$cos(\theta_{l})$')
    plt.ylabel('Relative efficiency')
    plt.title('8 < q^2 < 12')
    plt.grid()
    plt.show()

    phi = bin['phi']
    y_phi, x_phi = relative_histogram_generator(phi)
    popt_phi, cov_phi = curve_fit(n6_polynomial_even, x_phi, y_phi)
    plt.plot(x_phi, y_phi, '.', c='black')
    x = np.linspace(-np.pi, np.pi, 10000)
    plt.plot(x, n6_polynomial_even(x, *popt_phi), c='black')
    plt.xlabel(r'$\phi$')
    plt.ylabel('Relative efficiency')
    plt.title('8 < q^2 < 12')
    plt.grid()
    plt.show()

    #Save the fit parameters into array
    costhetak_popt_ls.append(popt_costhetak)
    costhetal_popt_ls.append(popt_costhetal)
    phi_popt_ls.append(popt_phi)
    costhetak_cov_ls.append(cov_costhetak)
    costhetal_cov_ls.append(cov_costhetal)
    phi_cov_ls.append(cov_phi)

#get total efficiency for each bin
#for bin in binned_data:
    #......
#%%
