import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy


def chisqr(obs, exp, error):
    chisqr = 0
    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i] - exp[i]) ** 2) / (error[i] ** 2)
    return chisqr


A_errs = np.loadtxt("A_errs.csv", delimiter=",", skiprows=1)
A_vals = np.loadtxt("A_vals.csv", delimiter=",", skiprows=1)
bin_ranges = [[.1, .98],
              [1.1, 2.5],
              [2.5, 4.0],
              [4.0, 6.0],
              [6.0, 8.0],
              [15., 17.],
              [17., 19.],
              [11., 12.5],
              [1.0, 6.0],
              [15., 19.]]

for column in range(1, A_vals[0, :].shape[0]):
    bin_ranges_means = [(i[0] + i[1]) / 2 for i in bin_ranges]
    bin_distances = [abs(i[0] - i[1]) for i in bin_ranges]

    plt.errorbar(bin_ranges_means, A_vals[:, column],
                 yerr=abs(A_errs[:, column]), xerr=bin_distances,
                 fmt="ro")
    plt.show()
    chi_sqr = chisqr(A_vals[:, column], np.zeros(A_vals[:, column].shape),
                     A_errs[:, column])
    degrees_of_freedom = A_vals[:, column].shape[0] - 1
    reduced_chi_sqr = chi_sqr / degrees_of_freedom
    p = 1 - scipy.stats.chi2.cdf(chi_sqr, degrees_of_freedom)
    print("Reduced Chi Squared for A_{} = {:.2f}".format(column + 2,
                                                         reduced_chi_sqr))
    print("p value = {:.4f}".format(p))

# now calculate the p value fitting all points to linear regression
all_vals = A_vals[:, 1:].flatten()
all_errs = A_errs[:, 1:].flatten()
chi_sqr = chisqr(all_vals, np.zeros(all_vals.shape[0]), all_errs)
degrees_of_freedom = all_vals.shape[0] - 1
reduced_chi_sqr = chi_sqr / degrees_of_freedom
p = 1 - scipy.stats.chi2.cdf(chi_sqr, degrees_of_freedom)
print("Reduced Chi Squared for all observ. = {:.2f}".format(reduced_chi_sqr))
print("p value = {:.4f}".format(p))

