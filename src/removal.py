from xgboost import Booster, DMatrix
import matplotlib.pyplot as plt
from comb_background_recogniser import load_path, EXPERIMENT

# Configure plotter
plt.style.use("ggplot")
plt.rcParams['figure.dpi'] = 300

# load in combinatorial background identifier
model = Booster()
model.load_model("comb_background_identifier.json")


df = load_path(EXPERIMENT) # total data

# Find Background
reduced_df = df[model.feature_names]  # reduced data (to features used in model)
reduced_df_DMatrix = DMatrix(reduced_df)  # convert to correct type
predictions = model.predict(reduced_df_DMatrix)  # predict 

# Filter
df_filtered = df  # dataframe which will store filtered data
df_filtered["is_background"] = predictions  # assign predictions to new row
# predictions closer to zero indicate that event is not from comb. background
# predictions closer to one indicate that event is from comb. background
THRESHOLD = 0.05  # higher threshold means more data is kept
df_filtered = df_filtered[df_filtered.is_background < THRESHOLD]

# Plot unfiltered and filtered masses of B0
plt.xlabel("B0 Mass")
plt.ylabel("Number of Events")
masses = df["B0_M"]
masses_filtered = df_filtered["B0_M"]
plt.hist(masses, bins=int(5e2), label="Total Data")
plt.hist(masses_filtered, bins=int(5e2), label="Removed Background")
plt.legend()
plt.show()

# Plot unfiltered and filtered momenta of B0
plt.xlabel("B0 Momentum")
plt.ylabel("Number of Events")
masses = df["B0_P"]
masses_filtered = df_filtered["B0_P"]
plt.hist(masses, bins=int(5e2), label="Total Data")
plt.hist(masses_filtered, bins=int(5e2), label="Removed Background")
plt.legend()
plt.show()
