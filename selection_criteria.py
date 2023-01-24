#%%
import os
import pandas as pd


def load_data(path: str):
    return pd.read_csv('TBPS - Spring 2023/'+path)

def filter_prob(x: str, y:str, thres: float, df):
    return df[df[x+'_ProbNN'+y] >= thres]
    #might need to add condition about ghost prob here once we understand what it is

def filter_IPCHI2_OWNPV(x: str, thres1: float, thres2, df):
    dof = 4
    #return df[df[x+'_IPCHI2_OWNPV'] > thres1 and df[x+'_IPCHI2_OWNPV']/dof < thres2]
    # I am not sure about the dof condition, the two conditions above seem quite contradictory to each other
    # if not, we can probably start by using this:
    return df[df[x+'_IPCHI2_OWNPV'] > thres1]

def filter_B0(IPCHI2_OWNPV_thres: float, B0_M_thres_low: float, B0_M_thres_high: float, DIRA_OWNPV_thres: float, FDCHI2_OWNPV_thres: float, ENDVERTEX_CHI2_thres: float, df):
    return df[
        df['B0_IPCHI2_OWNPV'] < IPCHI2_OWNPV_thres and
        df['B0_M'] > B0_M_thres_low and 
        df['B0_M'] < B0_M_thres_high and
        df['B0_DIRA_OWNPV'] < DIRA_OWNPV_thres and #not quite sure about this one as the criteria in the paper is given in mrad, but data file appears to be a percent
        df['B0_FDCHI2_OWNPV'] >  FDCHI2_OWNPV_thres 
        #and df['B0_ENDVERTEX_CHI2'] >  #not sure about this condition either
    ]

def filter_kstar0(Kstar_M_thres: float, vertexchi2_thres: float, FDCHI2_OWNPV_thres: float, df):
    return df[
        df['Kstar_M'] < Kstar_M_thres and
        df['Kstar_ENDVERTEX_CHI2'] < vertexchi2_thres and
        df['']
    ]


def main():
    data = load_data('phimumu.csv')

    X = ['mu_plus', 'mu_minus', 'K', 'Pi']
    Y = ['mu', 'mu', 'k', 'pi']

    #Probability filters
    thres = 0.8
    for i in range(0, len(X)):
        data = filter_prob(X[i], Y[i], thres, data)

    #IPCHI2 filters
    thres = [9, 5]
    for i in range(0, len(X)):
        data = filter_IPCHI2_OWNPV(X[i], *thres, data)

    #B0 filters
    thres = [16, 4850, 5780, 121 , 'xxx'] #xxx means i don't understand what goes here
    data = filter_B0(*thres, data)


    
    


# %%

