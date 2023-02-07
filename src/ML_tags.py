from model_tester import MODEL_FILES, EXPERIMENT, load_path
from xgboost import Booster, DMatrix
import pandas as pd

THRESHOLD = 0.95  # Threshold cutoff for is_signal classification for each model


def combinedMLtag(row, df, THRESHOLD):
    """
    spits out the final combined tag for a given row given row given a
    data frame of tags for each background
    """

    for criterion in df.columns:
        if row[criterion] < THRESHOLD:
            return 0
    return 1


if __name__ == '__main__':
    df = load_path(EXPERIMENT)  # total data
    tags_df = pd.DataFrame()  # dict to store tags of all the models
    for file in MODEL_FILES:
        # load in model
        model = Booster()
        model.load_model(file)

        # get tags
        reduced_df = df[model.feature_names]  # only features used in model
        reduced_df_DMatrix = DMatrix(reduced_df)  # convert to correct type
        tags = model.predict(reduced_df_DMatrix)  # make tags

        tags_df["is_signal" + file[:-16]] = tags  # add tags to dataframe

    tags_df["is_signal_combinedMLtag"] = tags_df.apply(
        lambda row: combinedMLtag(row, tags_df, THRESHOLD), axis=1)

    tags_df.to_pickle("./ML_tags_all_models.pkl")
    tags_df["is_signal_combinedMLtag"].to_pickle("./ML_only_final_combined_pkl")