import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from featurewiz import featurewiz

# Note: Using pickled data speeds up the process considerably
# To pickle your data use "pandas.DataFrame.to_pickle"

EXPERIMENT = "total_dataset.pkl"
SIGNAL = "sig.pkl"
PEAKING = [
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
]


def load_path(name):
    # load in a pickled data passing in the file name
    return pd.read_pickle("../pickled_data/" + name)


if __name__ == "__main__":

    # first we plot B_0 mass histogram from total data (experimental data)
    data = load_path(EXPERIMENT)  # load in data
    masses = data["B0_M"]  # take B0 mass column
    plt.hist(masses, bins=int(5e2))  # bin masses into histogram
    plt.show()

    # XGboost will need to see how a backgound row looks
    # masses above a certain threshold are part of combinatorial background
    background = data.query("B0_M > 5500")
    background["is_background"] = 1  # tag background rows with 1
    masses = background["B0_M"]  # take masses above the threshold
    plt.hist(masses, bins=int(5e2))  # look at mass histogram above threshold
    plt.show()

    # XGboost will need to see how a signal row looks
    signal = load_path(SIGNAL)
    signal["is_background"] = 0  # tag all signal rows by 0
    masses = signal["B0_M"]
    plt.hist(masses, bins=int(5e2))
    plt.show()

    # make a combined data frame with the signal and background 
    # Note: join="inner" is a command which deletes non-mutual columns
    df = pd.concat([background, signal], join="inner", ignore_index=True)
    df.astype(np.float32)  # convert to 32 bit for RAM purposes
    del signal, background # free up some RAM

    # pop off tags before ANOVA analysis
    is_background = df.pop("is_background")

    # Identify important features with ANOVA
    features, train_m = featurewiz(df, target="B0_M", corr_limit=0.7, 
                                   verbose=2)
    reduced_df = df[features]  # delete redundant features

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(reduced_df,
                                                        is_background,
                                                        test_size=0.2)

    # Train model
    model = XGBClassifier(n_estimators=30, learning_rate=0.5)
    model.fit(X_train, Y_train)

    # plot the 'importance' of each of the reduced features
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.show()

    # Test the accuracy of the model using test datasets
    Y_pred = model.predict(X_test)
    accuracy = 100 * accuracy_score(Y_test, Y_pred)
    print("Accuracy: {:.2f}%".format(accuracy))

    # Save model
    model.save_model("comb_background_identifier.json")
