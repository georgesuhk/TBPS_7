from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from background_identifier import load_path, files
import numpy as np
import pandas as pd
import os
import sys

if __name__ == '__main__':
    # NOTE: only change the values of variables in CAPS when using!
    # setup
    EXPERIMENT = "acceptance_mc.pkl"
    COMBINATORIAL_THRESHOLD = 0.995
    save_folder_name = "ML_results_combThresh" + str(COMBINATORIAL_THRESHOLD) \
                       + "_" + EXPERIMENT[:-4]
    tags_write_file = "ML_tags_combThresh" + str(COMBINATORIAL_THRESHOLD) \
                      + "_" + EXPERIMENT
    current_directory = os.getcwd()
    folder_directory = os.path.join(current_directory, save_folder_name)
    os.mkdir(folder_directory)
    sys.stdout = open(folder_directory + "/results_info.txt", "w")

    # Configure plotter
    plt.style.use("ggplot")

    # load in background identifier
    model_comb = XGBClassifier(n_estimators=200, learning_rate=0.25)
    model_peaking = XGBClassifier(n_estimators=200, learning_rate=0.25)
    model_comb.load_model("comb_background_identifier_nest200_lrate0.25.json")
    model_peaking.load_model("background_identifier_nest200_lrate0.25.json")

    df = load_path(EXPERIMENT)
    # Make combinatorial predictions on data
    reduced_df = df[model_comb.feature_names_in_]  # reduce columns
    comb_predictions = model_comb.predict_proba(reduced_df)[:, 1]
    df["label_comb"] = comb_predictions
    # Make peaking decay predictions on total data
    df_peaking_reduced = df[model_peaking.feature_names_in_]
    peaking_predictions = model_peaking.predict(df_peaking_reduced)
    df["label_peaking"] = peaking_predictions

    # Filter out combinatorial background
    df_nocomb = df  # dataframe which will store filtered data
    df_comb = df  # dataframe which will store the comb background
    df_nocomb = df_nocomb.query("label_comb > " + str(COMBINATORIAL_THRESHOLD))
    df_comb = df_comb.query("label_comb < " + str(COMBINATORIAL_THRESHOLD))

    # PLOT COMB DEMONSTRATION:

    # Plot B0 masses for all events and for only the non-combinatorial events
    plt.figure()
    plt.title("Demonstrating The Removal of Combinatorial Background")
    plt.xlabel("B0 Mass")
    plt.ylabel("Number of Events")
    masses = df["B0_M"]
    masses_filtered = df_nocomb["B0_M"]
    print("Number of total non-combinatorial background events:",
          len(masses_filtered))
    plt.hist(masses, bins=int(5e2), label="Total Experimental Data")
    plt.hist(masses_filtered, bins=int(5e2), label="After "
                                                   "combinatorial background "
                                                   "is removed")
    plt.legend()
    plt.savefig(folder_directory + "/demonstrating_comb_removal.png", dpi=800)

    # PLOT THE COMB BACKGROUND:

    plt.figure()
    plt.title("The Background that was Removed")
    plt.xlabel("B0 Mass")
    plt.ylabel("Number of Events")
    masses_filtered = df_comb["B0_M"]
    print("Number of total combinatorial background events:",
          len(masses_filtered))
    plt.hist(masses_filtered, bins=int(5e2), label="Combinatorial Background "
                                                   "Events")
    plt.legend()
    plt.savefig(folder_directory + "/the_removed_comb_background.png",
                dpi=800)

    # PLOT PEAKING BACKGROUNDS:

    # Plot the events remaining for each type of background
    for index, file in enumerate(files):
        # random colour for histogram
        c = [(np.random.uniform(), np.random.uniform(),
              np.random.uniform())]

        df_file = load_path(file)  # simulated data
        # Plot the simulated data
        plt.figure()
        plt.title(file[:-4] + " simulation")
        plt.xlabel("B0 Mass")
        plt.ylabel("Frequency")
        if index == 1 or index == 2:
            plt.hist(df_file["B0_M"], bins=500, color="k")
        else:
            plt.hist(df_file["B0_M"], bins=100, color="k")
        plt.savefig(folder_directory + "/simulated_" + file[:-4] + "_decay.png",
                    dpi=800)

        # Do filtering and plot corresponding decay extracted from data
        plt.figure()
        df_peaking_filtered = df.query("label_peaking == " + str(
            index))
        plt.xlabel("B0 Mass")
        plt.ylabel("Number of Events")
        plt.title("{} events in data".format(file[:-4]))
        masses = df_peaking_filtered["B0_M"]
        print("Number of {} events before comb removal:".format(file[:-4]),
              len(masses))
        if index == 1 or index == 2:
            plt.hist(masses, bins=int(5e2), color=c)
        else:
            plt.hist(masses, bins=int(1e2), color=c)
        plt.savefig(folder_directory + "/" + file[:-4] +
                    "_decays_found_in_data.png", dpi=800)

        # Do filtering and plot corresponding decay extracted from data
        plt.figure()
        df_nocomb_filtered = df.query("label_peaking ==" + str(index))
        df_nocomb_filtered = df_nocomb_filtered.query(
            "label_comb > " + str(COMBINATORIAL_THRESHOLD))
        plt.xlabel("B0 Mass")
        plt.ylabel("Number of Events")
        plt.title("{} events in data after comb background filtering".format(
            file[:-4]))
        masses = df_nocomb_filtered["B0_M"]
        print("Number of {} events after comb removal:".format(file[:-4]),
              len(masses))
        if index == 1 or index == 2:
            plt.hist(masses, bins=int(5e2), color=c)
        else:
            plt.hist(masses, bins=int(1e2), color=c)
        plt.savefig(folder_directory + "/after_comb_rem_" + file[:-4] +
                    "_decays_found_in_data.png", dpi=800)

    # Now simply extract tags for these parameters and dataset

    comb_tags = comb_predictions > COMBINATORIAL_THRESHOLD
    sig_tags = peaking_predictions == 0
    jpsi_tags = peaking_predictions == 1
    psi2s_tags = peaking_predictions == 2
    final_ML_tags = (sig_tags | jpsi_tags | psi2s_tags) & comb_tags

    pd.DataFrame(data={"is_sig_": final_ML_tags}).astype(
        int).to_pickle(folder_directory + "/" + tags_write_file)
