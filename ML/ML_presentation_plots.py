from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from background_identifier import load_path, files
import numpy as np
import pandas as pd
import os
import sys

EXPERIMENT = "total_dataset.pkl"

# Configure plotter
plt.style.use("ggplot")

# load in models
model_comb = XGBClassifier(n_estimators=200, learning_rate=0.25)
model_peaking = XGBClassifier(n_estimators=200, learning_rate=0.25)
model_comb.load_model("comb_background_identifier_nest200_lrate0.25.json")
model_peaking.load_model("background_identifier_nest200_lrate0.25.json")

# make a new dataset with label columns
df = load_path(EXPERIMENT)
# Make combinatorial predictions on data
reduced_df = df[model_comb.feature_names_in_]  # reduce columns
comb_predictions = model_comb.predict_proba(reduced_df)[:, 1]
df["label_comb"] = comb_predictions
# Make peaking decay predictions on total data
df_peaking_reduced = df[model_peaking.feature_names_in_]
peaking_predictions = model_peaking.predict(df_peaking_reduced)
peaking_predictions_proba = model_peaking.predict_proba(df_peaking_reduced)
df["label_peaking"] = peaking_predictions



# PLOT STUFF

# 1: Plot the total graph
bins = 100
plt.figure()
plt.xlabel("B0 Mass")
plt.ylabel("Number of Events")
data = df["B0_M"]
combinatorial_background = df[(df["label_comb"] < 0.75) &
                              (df["label_peaking"] < 3)]["B0_M"]
signal = df[(df["label_comb"] > 0.75) &
            (df["label_peaking"] == 0)]["B0_M"]
jpsi = df[(df["label_comb"] > 0.75) &
            (df["label_peaking"] == 1)]["B0_M"]
psi2s = df[(df["label_comb"] > 0.75) &
            (df["label_peaking"] == 2)]["B0_M"]
jpsi_mu_k_swap = df[(df["label_comb"] > 0.75) &
                (df["label_peaking"] == 3)]["B0_M"]
jpsi_mu_pi_swap = df[(df["label_comb"] > 0.75) &
                (df["label_peaking"] == 4)]["B0_M"]
phimumu = df[(df["label_comb"] > 0.75) &
             (df["label_peaking"] == 5)]["B0_M"]
Kmumu = df[(df["label_comb"] > 0.75) &
           (df["label_peaking"] == 6)]["B0_M"]
pKmumu_piTop = df[(df["label_comb"] > 0.75) &
                  (df["label_peaking"] == 7)]["B0_M"]
pKmumu_piTok_kTop =  df[(df["label_comb"] > 0.75) &
                        (df["label_peaking"] == 8)]["B0_M"]
k_pi_swap = df[(df["label_comb"] > 0.75) &
               (df["label_peaking"] == 9)]["B0_M"]
Kstarp_pi0 = df[(df["label_comb"] > 0.75) &
                (df["label_peaking"] == 10)]["B0_M"]
Jpsi_Kstarp_pi0 = df[(df["label_comb"] > 0.75) &
                     (df["label_peaking"] == 11)]["B0_M"]

#plt.hist(data, bins=bins, label="Total Experimental Data")
plt.hist(combinatorial_background, bins=bins,
         label="Combinatorial Background", histtype="step")
plt.hist(phimumu, bins=bins,
         label="phimumu", histtype="step")
plt.hist(k_pi_swap, bins=bins,
         label="k_pi_swap", histtype="step")
plt.hist(signal, bins=bins,
         label="Signal")
plt.hist(Jpsi_Kstarp_pi0, bins=bins,
         label="Jpsi_Kstarp_pi0", histtype="step")
plt.hist(Kmumu, bins=bins,
         label="Kmumu", histtype="step")
plt.hist(Kstarp_pi0, bins=bins,
         label="Kstarp_pi0", histtype="step")
plt.hist(jpsi_mu_pi_swap, bins=bins,
         label="jpsi_mu_pi_swap", histtype="step")
plt.hist(jpsi_mu_k_swap, bins=bins,
         label="jpsi_mu_k_swap", histtype="step")
plt.hist(pKmumu_piTop, bins=bins,
         label="pKmumu_piTop", histtype="step")
plt.hist(pKmumu_piTok_kTop, bins=bins,
         label="pKmumu_piTok_kTop", histtype="step")









plt.legend()
plt.show()








