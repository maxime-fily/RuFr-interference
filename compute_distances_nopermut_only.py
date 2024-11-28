##########################################################################
# modifications after august 2024 specifications (MFY) :
# 1. add a contextual name for the output file
import pickle
from pathlib import Path
from itertools import islice
import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from dtw import *

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", required=True, type=Path)
parser.add_argument("--output_dir", required=True, type=Path)
parser.add_argument("--repr_column", required=True)
parser.add_argument("--normalize", action="store_true")
parser.add_argument("--filt_W", required=True)

args = parser.parse_args()

print("loading data...")
df = pd.concat([pickle.load(open(filename, "rb"))
                for filename in tqdm(list(args.input_dir.glob("*.pkl")))], axis=0)
print(list(df))
# remove parts that are not annotated
# These correspond mainly to silence (and, sometimes, to parts with speech considered not relevant for the task)
# because I've asked myself the question and I don't want to plunge back into the abyss of thought: it's 
# important to keep the silences when aligning annotations ("words") and frames because the timestamps indicated in
# the annotations ("words") take these silences into account. But now that the alignment is done, time information is
# no longer important.
#
# XXX should we do that after having computed the normalized representations? XXX
df = df[df["words"] != ""]
df = df[df["sentence"].str.contains(str(args.filt_W))]


# compute normalized representations
# ----------------------------------
#
# The representations are normalized at corpus level, it would be interesting to see if it makes sense to normalize\
# them at a speaker level
# ATTENTION : there are two variables that are fairly different : repr_column and args.repr_column. The repr_column is\
# the one that includes normalization if the option is retained. args.repr_column is only an input data and NO CALC\
# whatsoever shall bedone with it.

print("normalize representations")
normalized_col = f"normalized_{args.repr_column}"
X = np.vstack(df[args.repr_column].values)

mu = X.mean(axis=0)
sigma = X.std(axis=0)

df[normalized_col] = df[args.repr_column].apply(lambda x: (x - mu) / sigma)

assert df.iloc[0][normalized_col].shape[0] == 1024

if args.normalize:
    suffnorm="NORM"
    repr_column = normalized_col
else:
    suffnorm="UNNO"
    repr_column = args.repr_column

print(f"using representations in {repr_column}")

# compute cross-product between words
# -----------------------------------
#
# A 1               A 1 2
# A 2               A 1 1
# B 1   ==>         A 2 2
# B 2                 ...

print("compute cross product")
merged_label = df.groupby(["filename", "speaker", "words", "uniq"])[repr_column]\
        .apply(lambda x: np.vstack(x))\
        .reset_index()

merged_label = pd.concat([group.merge(group, how="cross").assign(dupekey=lambda v: v[['filename_x', 'filename_y']]\
                        .apply(frozenset, axis=1)).drop_duplicates(subset=['dupekey'])\
                        .drop(columns=['dupekey'])
                        for _, group in merged_label.groupby("words")], axis=0)

def align(x, y):
    from scipy.spatial.distance import cdist
    
    dist = cdist(x, y, metric="cosine")
    return dtw(dist, keep_internals=True), dist

print("compute similarity")
sim = merged_label.apply(lambda row: align(row[repr_column + "_x"], 
                                           row[repr_column + "_y"]),
                        axis=1,
                        result_type="expand")
sim.columns = ["DTW","Cdist"]

final = pd.concat([sim, merged_label], axis=1)
args.output_dir.mkdir(exist_ok=True, parents=False)
os.chdir(args.output_dir)
print("writing to output...")
output_name = f"{args.filt_W}-{args.repr_column}-allspk-{suffnorm}.pkl"
final.to_pickle(output_name)
