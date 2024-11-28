import os
import statistics
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from dtw import *
import argparse
from numpy import unravel_index
import seaborn as sns
from frechetdist import frdist
from scipy.interpolate import interp1d
from itertools import cycle
from matplotlib import colormaps
import matplotlib.gridspec as gridspec
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True, type=Path)
parser.add_argument("--layer", required=True, type=int)
args = parser.parse_args()
from itertools import cycle

lines = ["-","--","-.",":","."]
linecycler = cycle(lines)
outsuf=str(args.input_file).split("-")[0]

def categorize(row,cx,cy):
    if row[cx] == row[cy]:
        return row[cx]  # If L1_x and L1_y are the same, return the language (FR/RU)
    else:
        return str(sorted([row[cx],row[cy]])[0]+" vs "+sorted([row[cx],row[cy]])[1])

os.chdir("/home/mfily/Documents/NOANOA_locallyowned_texfiles/COLING_2025")
df=pd.read_pickle(args.input_file)
df = df[~((df['spk_x'] == df['spk_y']) & (df['file_x'] == df['file_y']))]
print(list(df))
print(df["R"].head())
df['same_speaker'] = df['spk_x'] == df['spk_y']
df["L1"] = df.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df = df[df['layer'] == args.layer]
gs = gridspec.GridSpec(2, 6, height_ratios=[2, 1])
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[:, 1])
ax3 = fig.add_subplot(gs[:, 2])
ax7 = fig.add_subplot(gs[:, 3])
ax6 = fig.add_subplot(gs[:, 5])
ax4 = fig.add_subplot(gs[0, 4])
ax5 = fig.add_subplot(gs[1, 4])
ax1.set_title("audio_cost[-1;-1]")
ax2.set_title("vector_cost[-1;-1]")
ax3.set_title("Deviation")
ax4.set_title("dist_A")
ax5.set_title("dist_N")
ax6.set_title("Fréchet")
ax7.set_title("deviation length")
sns.boxplot(y='audio_cost_LD', data=df, x = 'same_speaker', ax = ax1, hue = 'L1')
sns.boxplot(y='cost_LD', data=df, x = 'same_speaker', ax = ax2, hue = 'L1')
sns.boxplot(y='R', data=df, x = 'same_speaker', ax = ax3, hue = 'L1')
sns.boxplot(y='dist_A', data=df, x = 'same_speaker', ax = ax4, hue = 'L1')
sns.boxplot(y='dist_N', data=df, x = 'same_speaker', ax = ax5, hue = 'L1')
sns.boxplot(y='frechet_dist', data=df, x = 'same_speaker', ax = ax6, hue = 'L1')
sns.boxplot(y='len_devxy', data=df, x = 'same_speaker', ax = ax7, hue = 'L1')
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
    ax.yaxis.set_label_text("")
fig.figure.savefig(f"{outsuf}_box_L1_hue_{args.layer}.pdf", bbox_inches='tight')
