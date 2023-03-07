"""
@author: Maxim Asmat
Date Created: Thursday 02/02/2023

This file generates a model which can classify an event as either part of
background or signal (labels 0 and 1 respectively). To train the model uses
sig, jpsi and psi2S simulation files as a proxy for a signal like event and
the uper mass range (B0 mass) of the total dataset of the experiment as a
proxy for a background. Notice that the upper mass range will contain
multiple peaking backgrounds as well as the combinatorial background. So
applying this model will remove the combinatorial background as well as
some peaking backgrounds (which are not of interest).

TO-DO:
- k-folding cross validation (get average accuracy and standard deviation)
- confusion matrix
- optimise parameters (n_estimators, learning_rate, tree_depth...)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from featurewiz import featurewiz
import time

# Files used
EXPERIMENT = "total_dataset.pkl"
SIGNALS = ["sig.pkl", "jpsi.pkl", "psi2S.pkl"]


def load_path(name):
    # load in a pickled data passing in the file name
    return pd.read_pickle("../pickled_data/" + name)


def extract_comb_background(EXPERIMENT):
    """
    given a path to total data, extract the combinatorial background as pd df
    :param EXPERIMENT: path to total data stored as the 0th elem in a list
    :return: pd dataframe of background data
    """
    return load_path(EXPERIMENT).query("B0_M > 5440")


if __name__ == '__main__':

    # XGboost will need to see how a background row looks
    # masses above a certain threshold are part of combinatorial background
    background = extract_comb_background(EXPERIMENT)
    background["is_sig"] = 0

    # XGboost will need to see how a signal row looks
    signal_dfs = [load_path(sig) for sig in SIGNALS]
    signal = pd.concat(signal_dfs, ignore_index=True)
    signal["is_sig"] = 1

    # make a combined data frame with the signal and background
    # Note: join="inner" is a command which deletes non-mutual columns
    df = pd.concat([background, signal], join="inner", ignore_index=True)

    df.astype(np.float32)  # convert to 32 bit for RAM purposes
    del signal, background  # free up some RAM

    is_sig = df["is_sig"]

    # find out which variables are correlated to B0_M to delete
    #a = df.corr().abs()["B0_M"]  # examine in debugger
    #b = df.corr().abs()  # examine in debugger

    # If entry in correlation matrix with B0_M was > 0.1 then del feature:
    df.pop("B0_M")  # Do not use B0_M as a feature when fitting
    df.pop("B0_DIRA_OWNPV") # feature correlated to B0_M
    df.pop("B0_ENDVERTEX_CHI2")  # feature correlated to B0_M
    df.pop("Kstar_ENDVERTEX_CHI2")  # feature correlated to B0_M

    # Identify important features with ANOVA/SULOV
    features, train_m = featurewiz(df, target="is_sig", corr_limit=0.9,
                                   verbose=2)
    # Model should not be skewed by B0_M
    reduced_df = df[features]  # delete redundant features

    del df  # save RAM

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(reduced_df,
                                                        is_sig,
                                                        test_size=0.2)

    # Train model
    model = XGBClassifier(n_estimators=200, learning_rate=0.25)
    model.fit(X_train, Y_train)

    # plot the 'importance' of each of the reduced features
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()

    # Test the accuracy of the model using test datasets
    Y_pred = model.predict(X_test)
    accuracy = 100 * accuracy_score(Y_test, Y_pred)
    print("Accuracy: {:.2f}%".format(accuracy))

    # Save model
    model.save_model("#2comb_background_identifier_nest200_lrate0.25.json")
