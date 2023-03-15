from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from background_identifier import load_path, files
import matplotlib.patches as mpatches
from matplotlib import ticker


EXPERIMENT = "total_dataset.pkl"


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

rcParams['savefig.bbox'] = 'tight'
#rcParams['savefig.dpi'] = 1200

rcParams['savefig.transparent'] = True
rcParams['figure.figsize'] = [16, 9]

params = {
    'savefig.bbox': 'tight',
    'savefig.dpi': 1000,
    'savefig.transparent': True,
    'figure.figsize': [18, 9],
    "xtick.top": False,
    "ytick.right": True,
    "xtick.direction": "out",
    "ytick.direction": "in",
}
rcParams.update(params)
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 15
# 1: Plot the total graph
alphas = [
    0.1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]
bins = 75
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

fig, ax1 = plt.subplots()

ax1.set_xlim(5170, 5970)
ax1.hist(jpsi, bins=bins, color="g", alpha=1)
ax1.hist(jpsi, bins=bins, lw=1, color="g",
         histtype="step")

ax1.hist(psi2s, bins=bins, color="#626262", alpha=1)
ax1.hist(psi2s, bins=bins, lw=1, color="#626262",
         histtype="step")


# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.31, 0.23, 0.57, 0.62]
ax2 = fig.add_axes([left, bottom, width, height])
ax1.indicate_inset([5170, 0, 530, 620], inset_ax=ax2, alpha=1, edgecolor="k")
ax2.set_xlim(5170, 5700)

# Combinatorial Background
ax1.hist(combinatorial_background, bins=bins, color="#d62728", alpha=alphas[0])
ax1.hist(combinatorial_background, bins=bins, lw=2, color="#d62728",
         histtype="step")

# k_pi_swap
ax1.hist(k_pi_swap, bins=bins,color="#ffe338", alpha=alphas[1])
ax1.hist(k_pi_swap, bins=bins, lw=1,
         color="#ffe338", histtype="step")

# phimumu
ax1.hist(phimumu, bins=bins, alpha=alphas[2], color="#80b9c7")
ax1.hist(phimumu, bins=bins, lw=1,
         color="#80b9c7", histtype="step")

# Jpsi_Kstarp_pi0
ax1.hist(Jpsi_Kstarp_pi0, bins=bins, alpha=alphas[4], color="#ff7f0e")
ax1.hist(Jpsi_Kstarp_pi0, bins=bins, lw=1,
         color="#ff7f0e", histtype="step")

# Kmumu
ax1.hist(Kmumu, bins=bins, alpha=alphas[5], color="#800080")
ax1.hist(Kmumu, bins=bins, lw=1,
         color="#800080", histtype="step")

# Kstarp_pi0
ax1.hist(Kstarp_pi0, bins=bins, alpha=alphas[6], color="#34a4fd")
ax1.hist(Kstarp_pi0, bins=bins, lw=1, color="#34a4fd", histtype="step")

# jpsi_mu_pi_swap
ax1.hist(jpsi_mu_pi_swap, bins=bins, alpha=alphas[7], color="#d4a4c4")
ax1.hist(jpsi_mu_pi_swap, bins=bins, lw=1,
         color="#d4a4c4", histtype="step")

# jpsi_mu_k_swap
ax1.hist(jpsi_mu_k_swap, bins=bins, alpha=alphas[8], color="#e377c2")
ax1.hist(jpsi_mu_k_swap, bins=bins, lw=1,
         color="#e377c2", histtype="step")

# pKmumu_piTop
ax1.hist(pKmumu_piTop, bins=bins, alpha=alphas[9], color="#c1d43f")
ax1.hist(pKmumu_piTop, bins=bins, lw=1,
         color="#c1d43f", histtype="step")

# pKmumu_piTok_kTop
ax1.hist(pKmumu_piTok_kTop, bins=bins, alpha=alphas[10], color="#9edae5")
ax1.hist(pKmumu_piTok_kTop, bins=bins, lw=1,
         color="#9edae5", histtype="step")

# Signal
ax1.hist(signal, bins=bins, alpha=alphas[3], color="#0072c6")
ax1.hist(signal, bins=bins, lw=1.5,
         color="k", histtype="step")

# Combinatorial Background
ax2.hist(combinatorial_background, bins=bins, color="#d62728", alpha=alphas[0])
ax2.hist(combinatorial_background, bins=bins, lw=2, color="#d62728",
         histtype="step")

# k_pi_swap
ax2.hist(k_pi_swap, bins=bins,color="#ffe338", alpha=alphas[1])
ax2.hist(k_pi_swap, bins=bins, lw=1,
         color="#ffe338", histtype="step", label="k_pi_swap")

# phimumu
ax2.hist(phimumu, bins=bins, alpha=alphas[2], color="#80b9c7")
ax2.hist(phimumu, bins=bins, lw=1,
         color="#80b9c7", histtype="step", label="phimumu")

# Jpsi_Kstarp_pi0
ax2.hist(Jpsi_Kstarp_pi0, bins=bins, alpha=alphas[4], color="#ff7f0e")
ax2.hist(Jpsi_Kstarp_pi0, bins=bins, lw=1,
         color="#ff7f0e", histtype="step", label="Jpsi_Kstarp_pi0")

# Kmumu
ax2.hist(Kmumu, bins=bins, alpha=alphas[5], color="#800080")
ax2.hist(Kmumu, bins=bins, lw=1,
         color="#800080", histtype="step")

# Kstarp_pi0
ax2.hist(Kstarp_pi0, bins=bins, alpha=alphas[6], color="#34a4fd")
ax2.hist(Kstarp_pi0, bins=bins, lw=1, color="#34a4fd", histtype="step",
         label="Kstarp_pi0")

# jpsi_mu_pi_swap
ax2.hist(jpsi_mu_pi_swap, bins=bins, alpha=alphas[7], color="#d4a4c4")
ax2.hist(jpsi_mu_pi_swap, bins=bins, lw=1,
         color="#d4a4c4", histtype="step", label="jpsi_mu_pi_swap")

# jpsi_mu_k_swap
ax2.hist(jpsi_mu_k_swap, bins=bins, alpha=alphas[8], color="#e377c2")
ax2.hist(jpsi_mu_k_swap, bins=bins, lw=1,
         color="#e377c2", histtype="step", label="jpsi_mu_k_swap")

# pKmumu_piTop
ax2.hist(pKmumu_piTop, bins=bins, alpha=alphas[9], color="#c1d43f")
ax2.hist(pKmumu_piTop, bins=bins, lw=1,
         color="#c1d43f", histtype="step", label="pKmumu_piTop")

# pKmumu_piTok_kTop
ax2.hist(pKmumu_piTok_kTop, bins=bins, alpha=alphas[10], color="#9edae5")
ax2.hist(pKmumu_piTok_kTop, bins=bins, lw=1,
         color="#9edae5", histtype="step", label="pKmumu_piTok_kTop")

# Signal
ax2.hist(signal, bins=bins, alpha=alphas[3], color="#0072c6")
ax2.hist(signal, bins=bins, lw=1.5,
         color="k", histtype="step")

handles1, labels = ax1.get_legend_handles_labels()
jpsi_patch = mpatches.Patch(facecolor="g", edgecolor="g", linewidth=1,
                            label="Jpsi")
psi2s_patch = mpatches.Patch(facecolor="#626262", edgecolor="#626262", linewidth=1,
                            label="Psi2s")
handles1.extend([jpsi_patch, psi2s_patch])

handles2, labels = ax1.get_legend_handles_labels()
comb_patch = mpatches.Patch(facecolor="#FBE8E9", edgecolor="#d62728",
                            linewidth=2,
                            label="Combinatorial Background")
sig_patch = mpatches.Patch(facecolor="#0072c6", edgecolor="k",
                           linewidth=1.5, label="Signal")
kmumu_patch = mpatches.Patch(facecolor="#800080", edgecolor="#800080",
                        linewidth=1, label="Kmumu")
jpsi_Kstarp_pi0_patch = mpatches.Patch(facecolor="#ff7f0e", edgecolor="#ff7f0e",
                                       linewidth=1, label="Jpsi_Kstarp_pi0")
k_pi_swap_patch = mpatches.Patch(facecolor="#ffe338", edgecolor="#ffe338",
                                 linewidth=1, label="K_pi_swap")
phimumu_patch = mpatches.Patch(facecolor="#80b9c7", edgecolor="#80b9c7",
                         linewidth=1, label="Phimumu")
Kstarp_pi0_patch = mpatches.Patch(facecolor="#34a4fd", edgecolor="#34a4fd",
                                      linewidth=1, label="Kstarp_pi0")
jpsi_mu_k_swap_patch = mpatches.Patch(facecolor="#e377c2", edgecolor="#e377c2",
                                      linewidth=1, label="Jpsi_mu_k_swap")
jpsi_mu_pi_swap_patch = mpatches.Patch(facecolor="#d4a4c4", edgecolor="#d4a4c4",
                                       linewidth=1, label="Jpsi_mu_pi_swap")
pKmumu_piTop_patch = mpatches.Patch(facecolor="#c1d43f", edgecolor="#c1d43f",
                                    linewidth=1, label="pKmumu_piTop")
pKmumu_piTok_kTop_patch = mpatches.Patch(facecolor="#9edae5",
                                         edgecolor="#9edae5",
                                         linewidth=1, label="pKmumu_piTok_kTop")
handles2.extend([comb_patch, sig_patch, kmumu_patch,
                 jpsi_Kstarp_pi0_patch, k_pi_swap_patch,
                 phimumu_patch, jpsi_mu_k_swap_patch,
                 jpsi_mu_pi_swap_patch, pKmumu_piTop_patch,
                 pKmumu_piTok_kTop_patch])

ax1.legend(loc="upper left", handles=handles1)
ax2.legend(loc="upper right", handles=handles2)
ax2.ticklabel_format(style='sci',scilimits=(0, 0), axis='y')
ax1.ticklabel_format(style='sci',scilimits=(0, 0), axis='y')
ax1.set_xlabel(r"$m(K^{+}\pi^{-}\mu^{+}\mu^{-})$   [MeV/c$^2}$]")
ax2.set_xlabel(r"$m(K^{+}\pi^{-}\mu^{+}\mu^{-})$   [MeV/c$^2$]")
ax2.set_ylabel("Number of Events")
ax1.set_ylabel("Number of Events")
ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
#ax2.set_yscale("log")
plt.savefig("data_decomposition.png")