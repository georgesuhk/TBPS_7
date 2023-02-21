#%%
#Written by Henry.

from Efficiency_legendre_fit import *
#from legendre_test import *

from tqdm import tqdm
#%%
import json
import numpy as np
import statistics as stats
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

#data = pd.read_csv('acceptance_mc.csv')
#preprocessed_data = preprocess_jit(data)
#total_kress_coeff = get_efficiency_coeff_kress(*preprocessed_data, mode='total_unweighted', i_max=5, j_max=6, m_max=5, n_max=4)
#get_efficiency_coeff_kress(costhetal_ls, costhetak_ls, phi_ls, q2_ls, i_max=5, j_max=5, m_max=5, n_max=5, mode='total_unweighted')

def get_bootstrap_params(data, i_max=5, j_max=6, m_max=5, n_max=4):
    list_of_c_coff = []
    for i in tqdm(range(500)):
        seed = np.random.randint(low=0, high=1000000)
        event = data.sample(frac=1, replace=True, random_state=seed)
        preprocessed_data = preprocess_jit(event)
        list_of_c_coff.append(get_efficiency_coeff_kress(*preprocessed_data, mode='total_unweighted', i_max=i_max, j_max=j_max, m_max=m_max, n_max=n_max))
    with open('list_of_c_coff.json', 'w') as outfile:
        json.dump(list_of_c_coff, outfile)
       
def acceptance_kress(costhetal, costhetak, phi, q2, i_max=5, j_max=6, m_max=5, n_max=4):
   
    preprocessed_data = preprocess_jit(data)
    total_kress_coeff = get_efficiency_coeff_kress(*preprocessed_data, mode='total_unweighted', i_max=5, j_max=6, m_max=5, n_max=4)
    efficiency = get_efficiency_kress(costhetal=costhetal, costhetak=costhetak, phi=phi, q2=q2, coeff_ls=total_kress_coeff, i_max=i_max, j_max=j_max, m_max=m_max, n_max=n_max)
   
    with open('list_of_c_coff.json') as file_:
        params = json.load(file_)
    eff = []
    for param in params:
        eff.append(get_efficiency_kress(costhetal=costhetal, costhetak=costhetak, phi=phi, q2=q2, coeff_ls=param, i_max=i_max, j_max=j_max, m_max=m_max, n_max=n_max))
    print(eff)
    error = stats.stdev(eff)
   
    return efficiency, error
# %%
