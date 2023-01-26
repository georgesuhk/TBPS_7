#%%
import numpy as np
import pandas as pd
# %%
#Load file as CSV
def load_file(path: str):
    return pd.read_csv(path)

def filter_single_prob(x: str, y:str, thres: float, df):
    return df[df[x+'_ProbNN'+y] >= thres]

def filter_prob(x_ls: list, y_ls: list, probability_thres, df):
    for i in range(0, len(x_ls)):
        data = filter_prob(x_ls[i], y_ls[i], probability_thres, df)
    return data

def filter_criteria(feature_ls: list, thres_ls: list, operator: list, df):
    """Filters all the columns provided in feature_ls according to the threshold
    provided in thres_ls, according to operator in operator_ls

    Example:
    feature_ls = ['height', 'weight']
    thres_ls = [180, 70]
    operator = ['>', '<']
    This keeps all rows with height > 180 and weight < 70.

    Args:
        feature_ls (list): List of features.
        thres_ls (list): List of thresholds.
        operator (list): List of operators.
        df (pd dataframe): The dataframe to be filterd.
    """
    for i, feature in enumerate(feature_ls[0]):
        if operator[i] == '<':
            df = df[df[feature] < thres_ls[i]]
        elif operator[i] == '<=':
            df = df[df[feature] <= thres_ls[i]]
        elif operator[i] == '>':
            df = df[df[feature] > thres_ls[i]]
        elif operator[i] == '>=':
            df = df[df[feature] >= thres_ls[i]]
        else:
            print(f'An invalid operator has been given for the feature {feature}. Not filtered this column.')


# %%
def main():
    data = load_file('Kstarp_pi0.csv')

    X = ['mu_plus', 'mu_minus', 'K', 'Pi']
    Y = ['mu', 'mu', 'k', 'pi']
    probability_thres = 0.8
    data_filtered = filter_prob(X, Y, probability_thres, data)

    feature_ls = ['mu_plus_IPCHI2_OWNPV', 'mu_minus_IPCHI2_OWNPV', 'K_IPCHI2_OWNPV', 'Pi_IPCHI2_OWNPV'
        'B0_IPCHI2_OWNPV', 'B0_M', 'B0_M', 'B0_DIRA_OWNPV', 'B0_FDCHI2_OWNPV',
        'Kstar_M', 'Kstar_FDCHI2_OWNPV',
        'J_psi_M'
    ]
    thres_ls = [
        9, 9, 9, 9,
        16, 4850, 5780, np.cos(14e-3), 121,
        6200, 16,
        7100
    ]
    operator_ls = [
        '>', '>', '>', '>',
        '<', '>', '<', '<', '>',
        '<', '>',
        '<'
    ]
    data_filtered = filter_criteria(feature_ls, thres_ls, operator_ls)
    return data_filtered

filtered_test = main()

    
# %%
