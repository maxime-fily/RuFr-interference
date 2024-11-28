# For each frame in the XLSR-53 representation, find the phone its belongs to
#
#
# Columns in the output dataframe
# - layer_i: np.array → a vector
# - time:
# - speaker: str → 
# - filename: str → the name of the file in which the utterance was stored
# - sentence: str → the sentence spoken in this utterance
# - start_time: float → 
# - end_time phone: float → 
#
# There are as many rows as there are frames in the corpus
##########################################################################
# modifications after august 2024 specifications (MFY) :
# 1. added 2 layers to the output
# 2. added a unique identifier
#

import pickle

from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm

input_dir = "/home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora/"
output_dir = Path("/home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora")

# ---- ANONYMOUS CODE ----


def align_representations(row,
                          label_column,
                          embedding_column,
                          freq=1 / 49):

    row = row.to_frame().T.reset_index().copy()

    # create a DataFrame with the words and their timestamp
    # there is one phone per row
    Words = row[[label_column]].explode(column=label_column)\
                                .apply(lambda x: {"start_time": x[label_column].minTime,
                                                  "end_time": x[label_column].maxTime,
                                                  "words": x[label_column].mark},
                                        axis=1,
                                        result_type="expand")

    Words["words"] = Words["words"].str.replace("\t", "")
    Words["uniq"] = Words["words"] + "_" + Words["start_time"].astype(str)
    # create a dataframe with the embedding
    # there is one embedding per line
    representations = row[[embedding_column]].explode(column=embedding_column)
    representations["time"] = freq
    representations["time"] = representations["time"].cumsum() - freq
    representations["speaker"] = row["speaker"]
    representations["filename"] = row["filename"]
    representations["sentence"] = row["sentence"]

#    representations["uniqID"] = Words["words"] + "_" + representations["time"].astype(str)

    return pd.merge_asof(representations, Words, left_on="time", right_on="start_time")


if __name__ == "__main__":

    for i in range(25):
        lay_name=f"layer_{i}"
        lay_dirname = f"{output_dir}/lay_{i}_aligned"
        lay_dir=Path(lay_dirname)
        lay_dir.mkdir(parents=True, exist_ok=True)
        input_file = f"{input_dir}RuFr_interfer_lay_{str(i)}.pkl"
        df = pickle.load(open(input_file, "rb"))
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            d = align_representations(row,
                                label_column="words",
                                embedding_column=lay_name,)
            output_filename = lay_dir / f"{d.iloc[0]['filename'].stem}_aligned.pkl"
            d.to_pickle(output_filename)
