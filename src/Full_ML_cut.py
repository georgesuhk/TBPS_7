from model_trainer import train_model, load_path
from tqdm import tqdm

# Data and savenames
TOTAL_DATA = "total_dataset.pkl"
SIGNAL = "sig.pkl"
BACKGROUNDS = [
    "jpsi.pkl"
    "jpsi_mu_k_swap.pkl",
    "psi2S.pkl",
    "jpsi_mu_pi_swap.pkl",
    "phimumu.pkl",
    "Kmumu.pkl",
    "pKmumu_piTop.pkl",
    "pKmumu_piTok_kTop.pkl",
    "k_pi_swap.pkl",
    "Kstarp_pi0.pkl",
    ]
MODEL_NAMES = [
    "jpsi_mu_k_swap_Identifier",
    "psi2S_Identifier",
    "jpsi_mu_pi_swap_Identifier",
    "phimumu_Identifier",
    "Kmumu_Identifier",
    "pKmumu_piTop_Identifier",
    "pKmumu_piTok_kTop_Identifier",
    "k_pi_swap_Identifier",
    "Kstarp_pi0_Identifier",
]

if __name__ == '__main__':
    # make a model specifically for comb background
    train_model(TOTAL_DATA, SIGNAL, "comb_background_Identifier",
                comb_back=True)

    # make a model for each background to be removed
    for index, background in enumerate(tqdm(BACKGROUNDS)):
        train_model(background, SIGNAL, MODEL_NAMES[index])