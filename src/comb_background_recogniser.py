import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier, DMatrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from featurewiz import featurewiz

# Note: Using pickled data speeds up the process considerably
# To pickle your data use "pandas.DataFrame.to_pickle"

EXPERIMENT = "total_dataset.pkl"
SIGNAL = "jpsi.pkl"


def load_path(name):
    # load in a pickled data passing in the file name
    return pd.read_pickle("../pickled_data/" + name)


if __name__ == "__main__":

    # XGboost will need to see how a background row looks
    # masses above a certain threshold are part of combinatorial background
    background = load_path(EXPERIMENT).query("B0_M > 5375") # take masses above the threshold
    background["is_signal"] = 0

    # XGboost will need to see how a signal row looks
    signal = load_path(SIGNAL)
    signal["is_signal"] = 1

    # make a combined data frame with the signal and background
    # Note: join="inner" is a command which deletes non-mutual columns
    df = pd.concat([background, signal], join="inner", ignore_index=True)

    """
    # Set peaking files to background too
    for file in PEAKING:
        df_temp = load_path(file)
        df_temp["is_signal"] = 1  # tag peaking files as background
        df = pd.concat([df, df_temp], join="inner", ignore_index=True)
    """

    df.astype(np.float32)  # convert to 32 bit for RAM purposes
    del signal, background  # free up some RAM

    # pop off tags before ANOVA/SULOV analysis
    is_signal = df.pop("is_signal")

    # Identify important features with ANOVA/SULOV
    features, train_m = featurewiz(df, target="B0_M", corr_limit=0.7,
                                   verbose=2)
    reduced_df = df[features]  # delete redundant features

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(reduced_df,
                                                        is_signal,
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
    model.save_model("comb_background_Identifier.json")



"""PEAKING = [
    "jpsi_mu_k_swap.pkl",
    "psi2S.pkl",
    "jpsi_mu_pi_swap.pkl",
    "phimumu.pkl",
    "Kmumu.pkl",
    "pKmumu_piTop.pkl",
    "pKmumu_piTok_kTop.pkl",
    "k_pi_swap.pkl",
    "jpsi.pkl",
    "Kstarp_pi0.pkl",
]"""
