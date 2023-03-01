from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from background_identifier import load_path, files
import numpy as np
import pandas as pd


if __name__ == '__main__':
    TOTAL_DATA = "total_dataset.pkl"
    ACCEPTANCE = "acceptance_mc.pkl"

    # Calculate N_sig / N_JpsiK*

    peaking_model = XGBClassifier(n_estimators=200, learning_rate=0.25)
    comb_model = XGBClassifier(n_estimators=200, learning_rate=0.25)
    peaking_model.load_model("background_identifier_nest200_lrate0.25.json")
    comb_model.load_model("comb_background_identifier_nest200_lrate0.25.json")

    df = load_path(TOTAL_DATA)
    print(type(df))
    reduced_peaking_df = df[peaking_model.feature_names_in_]
    reduced_comb_df = df[comb_model.feature_names_in_]
    decay_predictions = peaking_model.predict(reduced_peaking_df)
    comb_predictions = comb_model.predict_proba(reduced_comb_df)[:, 1]
    del reduced_peaking_df, reduced_comb_df
    df["peaking_label"] = decay_predictions
    df["comb_label"] = comb_predictions

    q2_bins = [
        (0.1, 0.98),
        (1.1, 2.5),
        (2.5, 4.0),
        (4.0, 6.0),
        (6.0, 8.0),
        (15.0, 17.0),
        (17.0, 19.0),
        (11.0, 12.5),
        (1.0, 6.0),
        (15.0, 19.0),
    ]

    N_sigs = []
    N_JpsiKstars = []

    for bin in q2_bins:
        df_q2_sig = df[(df["q2"] > bin[0])
                       & (df["q2"] < bin[1])
                       & (df["comb_label"] > 0.95)
                       & (df["peaking_label"] == 0)]
        df_q2_JpsiKstar = df[(df["q2"] > bin[0])
                             & (df["q2"] < bin[1])
                             & (df["comb_label"] > 0.95)
                             & (df["peaking_label"] == 1)]
        N_sig = df_q2_sig.shape[0]
        N_JpsiKstar = df_q2_JpsiKstar.shape[0]
        print([N_sig, N_JpsiKstar])

        N_sigs.append(N_sig)
        N_JpsiKstars.append(N_JpsiKstar)

    N_sigs = np.array(N_sigs)
    N_JpsiKstars = np.array(N_JpsiKstars)

    print(sum(N_sigs), sum(N_JpsiKstars))
    print(N_sigs / (N_JpsiKstars + N_sigs))

    # Get Jpsi events in acceptance_mc.csv

    df_acceptance = load_path(ACCEPTANCE)




