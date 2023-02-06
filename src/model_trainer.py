import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from featurewiz import featurewiz
import time


def load_path(name):
    # load in a pickled data passing in the file name
    return pd.read_pickle("../pickled_data/" + name)


def train_model(BACKGROUND, SIGNAL, save_name, comb_back=False):
    if comb_back:
        # for comb. background the background is only part of tot. data.
        background = load_path(BACKGROUND).query("B0_M > 5375")
    else:
        background = load_path(BACKGROUND)  # load in what you want to remove
    background["is_signal"] = 0  # background is not signal

    signal = load_path(SIGNAL)  # load in what you want to keep
    signal["is_signal"] = 1  # signal is signal

    # make a combined data frame with the signal and background
    # Note: join="inner" is a command which deletes non-mutual columns
    df = pd.concat([background, signal], join="inner", ignore_index=True)

    df.astype(np.float32)  # convert to 32 bit for RAM purposes
    del signal, background  # free up some RAM

    # pop off tags before ANOVA/SULOV analysis
    is_signal = df.pop("is_signal")

    # Identify important features with ANOVA/SULOV
    features, train_m = featurewiz(df, target="B0_M", corr_limit=0.7)
    reduced_df = df[features]  # delete redundant features

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(reduced_df,
                                                        is_signal,
                                                        test_size=0.2)

    # Train model
    start_time = time.time()
    model = XGBClassifier(n_estimators=200, learning_rate=0.3)
    model.fit(X_train, Y_train)
    fitting_time = time.time() - start_time
    print("Model {} took {:.2f}s to fit data".format(save_name, fitting_time))

    # Test the accuracy of the model using test datasets
    Y_pred = model.predict(X_test)
    accuracy = 100 * accuracy_score(Y_test, Y_pred)
    print("Accuracy of model {}: {:.2f}%".format(save_name, accuracy))

    # Save model
    model.save_model(save_name + ".json")
