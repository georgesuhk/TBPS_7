#%%
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# %%
#Load file as CSV
def load_file(path: str):
    return pd.read_csv(path)

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
    for i, feature in enumerate(feature_ls):
        
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
    return df



def examine_filters(feature_ls: list, thres_ls: list, operator: list, df):
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
    Return a dataframe with the percentage of data remaining.
    """
    df_len = len(df)
    ratios = []
    for i, feature in enumerate(feature_ls):
        if operator[i] == '<':
            filtered_df_len = len(df[df[feature] < thres_ls[i]])
        elif operator[i] == '<=':
            filtered_df_len = len(df[df[feature] <= thres_ls[i]])
        elif operator[i] == '>':
            filtered_df_len = len(df[df[feature] > thres_ls[i]])
        elif operator[i] == '>=':
            filtered_df_len = len(df[df[feature] >= thres_ls[i]])
        else:
            print(f'An invalid operator has been given for the feature {feature}. Not filtered this column.')
        ratios.append(filtered_df_len/df_len)
    return ratios

def tag(feature_ls: list, thres_ls: list, operator: list, df):
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
    
    for i, feature in enumerate(feature_ls):
        
        if operator[i] == '<':
            df[i] = df[feature] < thres_ls[i]
        elif operator[i] == '<=':
            df[i] = df[feature] <= thres_ls[i]
        elif operator[i] == '>':
            df[i] = df[feature] > thres_ls[i]
        elif operator[i] == '>=':
            df[i] = df[feature] >= thres_ls[i]
        else:
            print(f'An invalid operator has been given for the feature {feature}. Not filtered this column.')
    #print(df[list(range(len(feature_ls)))].all(axis='columns').map({False: 0, True: 1}))
    df['is_signal_selection'] = df[list(range(len(feature_ls)))].all(axis='columns').map({False: 0, True: 1})
    df.drop(columns=list(range(len(feature_ls))), inplace=True)
    print('{} % of the initial data sized: {} was kept after the filtering'.format(df['is_signal_selection'].sum()/len(df), len(df)))
    return df


# %%
def filter_dataset(data):
    feature_ls = [#'mu_plus_ProbNNmu', 'mu_minus_ProbNNmu', 'K_ProbNNk', 'Pi_ProbNNpi',
        'mu_plus_IPCHI2_OWNPV', 'mu_minus_IPCHI2_OWNPV', 'K_IPCHI2_OWNPV', 'Pi_IPCHI2_OWNPV',
        'B0_IPCHI2_OWNPV', 'B0_M', 'B0_M', 'B0_FDCHI2_OWNPV', #'B0_DIRA_OWNPV', 
        'Kstar_M', 'Kstar_FDCHI2_OWNPV',
        'J_psi_M'
    ]
    thres_ls = [#0.8, 0.8, 0.8, 0.8,
        9, 9, 9, 9,
        16, 4850, 5780, 121, #np.cos(14e-3), 
        6200, 16,
        7100
    ]
    operator_ls = [#'>', '>', '>', '>',
        '>', '>', '>', '>',
        '<', '>', '<', '>', #'<', 
        '<', '>',
        '<'
    ]
    #data_filtered = filter_criteria(feature_ls, thres_ls, operator_ls, data)
    #return data_filtered
    return tag(feature_ls, thres_ls, operator_ls, data)
    
    


def main():
    feature_ls = [#'mu_plus_ProbNNmu', 'mu_minus_ProbNNmu', 'K_ProbNNk', 'Pi_ProbNNpi',
        'mu_plus_IPCHI2_OWNPV', 'mu_minus_IPCHI2_OWNPV', 'K_IPCHI2_OWNPV', 'Pi_IPCHI2_OWNPV',
        'B0_IPCHI2_OWNPV', 'B0_M', 'B0_M', 'B0_DIRA_OWNPV', 'B0_FDCHI2_OWNPV',
        'Kstar_M', 'Kstar_FDCHI2_OWNPV',
        'J_psi_M'
    ]
    
    df = load_file('acceptance_mc.csv')
    filter_dataset(df)['is_signal_selection'].to_csv('acceptance_post_stripping_cut.csv')
    
    #files = os.listdir('./data/')
    #list_of_ratios = [filter_dataset(load_file(file_name)) for file_name in tqdm(files)]
    #pd.DataFrame(list_of_ratios, columns = feature_ls, index = files).to_csv('filter_statistics.csv')


# %%
