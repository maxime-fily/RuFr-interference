# Create a pickled dataframe from the original corpus
# 
# create the files `output_filename` which contain the representations on all layers and 
# `output_small_filename` which only contains the representations on the last layers.
#
# Columns of the dataframes:
# filename: name of the file from which the data have been extracted (can be used as 
#           an id of the sentence/utterance)
# speaker: name of the speaker
# words: annotations at the word level
# phones: annotations at the phone level
# sentence: the sentence spoken
# layer_0', ..., 'layer_24' → representation on the i-th layer

import pickle
import io

from pathlib import Path

import torch
import pandas as pd
import textgrid

from tqdm import tqdm


corpus_dir = "../FRcorp"
output_filename = "/home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora/RuFr_interference.pkl"
output_small_filename = "/home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora/RuFr_interference_small.pkl"

# ---- ANONYMOUS CODE ----

def read_txt(filename):
    print(filename.with_suffix(".txt"))
    with open(filename.with_suffix(".txt"), "r") as ofile:
        return ofile.read().strip()

    
def read_representations(wav_filename):

    # the representation have been pickled without having been detached from the gpu
    # we need the following class to read them on a computer without gpu
    # see: https://github.com/pytorch/pytorch/issues/16797
    class CPU_Unpickler(pickle.Unpickler):

        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else: return super().find_class(module, name)

    base_name = wav_filename.with_suffix("")
    print(base_name)
    return {f"layer_{i}": CPU_Unpickler(open(f"{base_name}_{i:02}.pkl", "rb")).load().numpy() 
                        for i in range(0, 25)}


def read_textgrid(filename):
    t = textgrid.TextGrid.fromFile(filename.with_suffix(".TextGrid"))

    return {name: tiers for name, tiers in zip(t.getNames(), t.tiers)}

    
df = pd.DataFrame({"filename": Path(corpus_dir).rglob("*.wav")})
df["speaker"] = df["filename"].apply(lambda x: x.parent.stem)

assert df.shape[0] != 0, "did not find corpus"

df = pd.concat([df, df.apply(lambda x: read_representations(x["filename"]),
                                axis=1,
                                result_type="expand")],
                axis=1)

df = pd.concat([df, df.apply(lambda x: read_textgrid(x["filename"]),
                                axis=1,
                                result_type="expand")],
                axis=1)

df["sentence"] = df["words"].apply(lambda x: " ".join(interval.mark for interval in x if interval.mark))
# remove \t which correspond to input errors by annotators
df["sentence"] = df["sentence"].apply(lambda x: " ".join(x.split()))

df.to_pickle(output_filename)

df[['filename', 'speaker', 'layer_24', 'sentence', 'words', 'phones']].to_pickle(output_small_filename)
