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
from matplotlib.ticker import MultipleLocator, FuncFormatter
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True, type=Path)
parser.add_argument("--layer", required=True, type=int)
args = parser.parse_args()
from itertools import cycle
def categorize(row,cx,cy):
    if row[cx] == row[cy]:
        return row[cx]  # If L1_x and L1_y are the same, return the language (FR/RU)
    else:
        return str(sorted([row[cx],row[cy]])[0]+" vs "+sorted([row[cx],row[cy]])[1])

def format_ticks(value, tick_number):
    return f"{value:.1f}"
major_tick_interval = 0.1

lines = ["-","--","-.",":","."]
linecycler = cycle(lines)
outsuf=str(args.input_file).split("-")[0]
normopt=str(args.input_file).split("-")[1].replace(".pkl","")
os.chdir("/home/mfily/Documents/NOANOA_locallyowned_texfiles/COLING_2025")
df=pd.read_pickle(args.input_file)
c_it=['KL', 'KV', 'MD', 'AN', 'SD', 'YC', 'UV', 'RN']
df=df[((df['spk_x'] == c_it[0])|(df['spk_x'] == c_it[1])|(df['spk_x'] == c_it[2])|(df['spk_x'] == c_it[3])|(df['spk_x'] == c_it[4])|(df['spk_x'] == c_it[5])|(df['spk_x'] == c_it[6])|(df['spk_x'] == c_it[7]))&\
        ((df['spk_y'] == c_it[0])|(df['spk_y'] == c_it[1])|(df['spk_y'] == c_it[2])|(df['spk_y'] == c_it[3])|(df['spk_y'] == c_it[4])|(df['spk_y'] == c_it[5])|(df['spk_x'] == c_it[6])|(df['spk_y'] == c_it[7]))]
df = df[~((df['spk_x'] == df['spk_y']) & (df['file_x'] == df['file_y']))]
print(list(df))
print(df["layer"].unique())

df['same_speaker'] = df['spk_x'] == df['spk_y']
df["L1"] = df.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df["AGAB"] = df.apply(categorize, axis=1, cx='sex_x', cy='sex_y')
df = df[df['layer'] == args.layer]
valid_combinations1 = df[~((df['AGAB'] == 'f vs m') & (df['same_speaker'] == True))]
valid_combinations2 = df[~((df['L1'] == 'fr vs ru') & (df['same_speaker'] == True))]
fig = plt.figure(figsize=(6, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
#fig.suptitle("vector_cost_norm[-1;-1]")
ax1.yaxis.set_major_locator(MultipleLocator(major_tick_interval))
ax2.yaxis.set_major_locator(MultipleLocator(major_tick_interval))
ax1.yaxis.set_major_formatter(FuncFormatter(format_ticks))
ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks))

sns.boxplot(x='same_speaker', y='Ncost_LD', data=valid_combinations1, hue = 'AGAB', ax = ax1, dodge=True, hue_order=['f','m','f vs m'])
sns.boxplot(x='same_speaker', y='Ncost_LD', data=valid_combinations2, hue = 'L1', ax = ax2, dodge=True, hue_order=['fr','ru','fr vs ru'])

for ax in [ax1, ax2]:
    ax.yaxis.set_label_text("")
ax1.xaxis.set_label_text("")
fig.figure.savefig(f"{outsuf}_finalbox_{args.layer}_{normopt}_8SPK.pdf", bbox_inches='tight')
plt.clf()

