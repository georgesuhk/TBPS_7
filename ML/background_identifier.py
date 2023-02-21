"""
@author: Maxim Asmat
Date Created: Thursday 02/02/2023

This file generates a model which can identify the type of a certain decay,
using the 12 training classes given to it (these are the simulation files we
were provided with). This is done by tagging each decay by its type/class and
using XGBClassifier to train a model to identify each class. Before training
we identify features that are important to the type of decay using featurewiz
and delete redundant features. The model is then saved as a .json file and
can be applied to identify the type/class of a previously unseen decay event.

The model takes approximately 100 minutes to create.

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

# NOTE: MORE THAN 100 RUNTIME MINUTES TO CREATE THIS MODEL

# Note: Using pickled data speeds up the process considerably
# To pickle your data use "pandas.DataFrame.to_pickle"

# Names of all used data files
files = [  # the label number of each file is on the right
    "sig.pkl",  # 0
    "jpsi.pkl",  # 1
    "psi2S.pkl",  # 2
    "jpsi_mu_k_swap.pkl",  # 3
    "jpsi_mu_pi_swap.pkl",  # 4
    "phimumu.pkl",  # 5
    "Kmumu.pkl",  # 6
    "pKmumu_piTop.pkl",  # 7
    "pKmumu_piTok_kTop.pkl",  # 8
    "k_pi_swap.pkl",  # 9
    "Kstarp_pi0.pkl",  # 10
    "Jpsi_Kstarp_pi0.pkl",  # 11
]


def load_path(name):
    # load in a pickled data passing in the file name
    return pd.read_pickle("../pickled_data/" + name)


if __name__ == "__main__":

    # label all row with correct decay category
    df = pd.DataFrame()
    for index, file in enumerate(files):
        df_temp = load_path(file)
        df_temp["label"] = index
        df = pd.concat([df, df_temp], ignore_index=True)
        del df_temp  # save RAM by deleting temporary storage variable

    df.astype(np.float32)  # convert to 32 bit for RAM purposes

    classes = df["label"]  # save classes column for training

    # Identify important features with ANOVA/SULOV
    features, train_m = featurewiz(df, target="label", corr_limit=0.7,
                                   verbose=2)
    reduced_df = df[features]  # delete redundant features

    del df  # save RAM

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(reduced_df,
                                                        classes,
                                                        test_size=0.2)

    # Train model
    model = XGBClassifier(n_estimators=200, learning_rate=0.25)
    start_time = time.time()
    model.fit(X_train, Y_train)
    end_time = time.time() - start_time
    print("Time to do fitting was {:.2f}s)".format(end_time))

    # plot the 'importance' of each of the reduced features
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()

    # Test the accuracy of the model using test datasets
    Y_pred = model.predict(X_test)
    accuracy = 100 * accuracy_score(Y_test, Y_pred)
    print("Accuracy: {:.2f}%".format(accuracy))

    # Save model
    model.save_model("background_identifier_nest200_lrate0.25.json")
