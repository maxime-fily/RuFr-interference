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
os.chdir("/home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora")
df=pd.read_pickle('RuFr_interfer_lay_0.pkl')
a=0
for i, row in df.iterrows():
    tier=row["words"]

    for inter in tier.intervals:
        if inter.mark.replace("\t","") !="":
            print(inter.mark, inter.minTime, inter.maxTime)
            a=a+(inter.maxTime-inter.minTime)
print(a)
