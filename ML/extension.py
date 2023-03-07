from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from background_identifier import load_path, files
import numpy as np
import os
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
    reduced_peaking_df = df[peaking_model.feature_names_in_]
    reduced_comb_df = df[comb_model.feature_names_in_]
    decay_predictions = peaking_model.predict(reduced_peaking_df)
    comb_predictions = comb_model.predict_proba(reduced_comb_df)[:, 1]
    del reduced_peaking_df, reduced_comb_df
    df["peaking_label"] = decay_predictions
    df["comb_label"] = comb_predictions


    # Bins from mitesh
    """q2_bins = [
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
    ]"""

    # Bins from paper (to compare)
    q2_bins = [
            (0.10, 2.00),
            (2.00, 4.30),
            (4.30, 8.63),
            (10.09, 12.86),
            (14.18, 16.00),
            (16.00, 19.00),
            (1.00, 6),
        ]

    N_sigs = []

    for bin in q2_bins:
        df_q2_sig = df[(df["q2"] > bin[0])
                       & (df["q2"] < bin[1])
                       & (df["comb_label"] > 0.95)
                       & (df["peaking_label"] == 0)]
        N_sig = df_q2_sig.shape[0]
        N_sigs.append(N_sig)

    df_q2_JpsiKstar = df[(df["q2"] > 8)
                         & (df["q2"] < 14)
                         & (df["comb_label"] > 0.95)
                         & (df["peaking_label"] == 1)]

    N_JpsiKstar = df_q2_JpsiKstar.shape[0]

    N_sigs = np.array(N_sigs)

    print(N_sigs)
    print(sum(N_sigs), N_JpsiKstar)
    print(N_sigs / N_JpsiKstar)

    # Get Jpsi events in acceptance_mc.csv

    df_acceptance = load_path(ACCEPTANCE)

    # Now get the jpsi events from acceptance_mc.pkl

    df_acceptance = load_path(ACCEPTANCE)
    reduced_peaking_df_acceptance = df_acceptance[
        peaking_model.feature_names_in_]
    reduced_comb_df_acceptance = df_acceptance[
        comb_model.feature_names_in_]
    decay_predictions_acceptance = peaking_model.predict(
        reduced_peaking_df_acceptance)
    comb_predictions_acceptance = comb_model.predict_proba(
        reduced_comb_df_acceptance)[:, 1]
    del reduced_peaking_df_acceptance, reduced_comb_df_acceptance
    df_acceptance["peaking_label"] = decay_predictions_acceptance
    df_acceptance["comb_label"] = comb_predictions_acceptance

    df_acceptance = df_acceptance[(df_acceptance["peaking_label"] == 2)
                                  & (df_acceptance["comb_label"] > 0.95)]
    plt.hist(df_acceptance["B0_M"], bins=100)
    plt.show()

    current_dir = os.getcwd()

    df_acceptance.to_pickle(os.path.join(
        current_dir +"/extension_acceptance_JpsiOnlyFilter_0.95Thresh.pkl"))





