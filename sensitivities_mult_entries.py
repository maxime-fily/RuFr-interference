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
parser.add_argument("--input_file1", required=True, type=Path)
parser.add_argument("--input_file2", required=True, type=Path)
parser.add_argument("--input_file3", required=True, type=Path)
parser.add_argument("--input_file4", required=True, type=Path)
parser.add_argument("--input_file5", required=True, type=Path)
parser.add_argument("--input_file6", required=True, type=Path)
parser.add_argument("--layer", required=True, type=int)
args = parser.parse_args()
from itertools import cycle

lines = ["-","--","-.",":","."]
linecycler = cycle(lines)
outsuf="multiword"
os.chdir("/home/mfily/Documents/NOANOA_locallyowned_texfiles/COLING_2025")
df1=pd.read_pickle(args.input_file1)
df2=pd.read_pickle(args.input_file2)
df3=pd.read_pickle(args.input_file3)
df4=pd.read_pickle(args.input_file4)
df5=pd.read_pickle(args.input_file5)
df6=pd.read_pickle(args.input_file6)
df=pd.concat([df1,df2,df3,df4,df5,df6], axis=0)
df = df[~((df['spk_x'] == df['spk_y']) & (df['file_x'] == df['file_y']))]
print(list(df))
print(df["layer"].unique())
df['same_speaker'] = df['spk_x'] == df['spk_y']
df = df[df['layer'] == args.layer]
gs = gridspec.GridSpec(2, 6, height_ratios=[2, 1])
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[:, 1])
ax3 = fig.add_subplot(gs[:, 2])
ax7 = fig.add_subplot(gs[:, 3])
ax6 = fig.add_subplot(gs[:, 5])
ax4 = fig.add_subplot(gs[0, 4])
ax5 = fig.add_subplot(gs[1, 4])
ax1.set_title("aud_cost_N[-1;-1]")
ax2.set_title("vec_cost_N[-1;-1]")
ax3.set_title("dev. rat.")
ax4.set_title("dist_A")
ax5.set_title("dist_N")
ax6.set_title("Fréchet")
ax7.set_title("Max grad")
#sns.boxplot(y='Naudio_cost_LD', data=df, hue = 'same_speaker', ax = ax1)
#sns.boxplot(y='Ncost_LD', data=df, hue = 'same_speaker', ax = ax2)
#sns.boxplot(y='R', data=df, hue = 'same_speaker', ax = ax3)
#sns.boxplot(y='dist_A', data=df, hue = 'same_speaker', ax = ax4)
#sns.boxplot(y='dist_N', data=df, hue = 'same_speaker', ax = ax5)
#sns.boxplot(y='frechet_dist', data=df, hue = 'same_speaker', ax = ax6)
#sns.boxplot(y='len_devxy', data=df, hue = 'same_speaker', ax = ax7)

sns.violinplot(y='Naudio_cost_LD', data=df, hue = 'same_speaker', ax = ax1)
sns.violinplot(y='Ncost_LD', data=df, hue = 'same_speaker', ax = ax2)
sns.violinplot(y='R', data=df, hue = 'same_speaker', ax = ax3)
sns.violinplot(y='dist_A', data=df, hue = 'same_speaker', ax = ax4)
sns.violinplot(y='dist_N', data=df, hue = 'same_speaker', ax = ax5)
sns.violinplot(y='frechet_dist', data=df, hue = 'same_speaker', ax = ax6)
sns.violinplot(y='Grad_Norm', data=df, hue = 'same_speaker', ax = ax7)
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
    ax.yaxis.set_label_text("")
fig.figure.savefig(f"{outsuf}_violin_{args.layer}.pdf", bbox_inches='tight')
plt.clf()

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[:, 1])
ax3 = fig.add_subplot(gs[:, 2])
ax7 = fig.add_subplot(gs[:, 3])
ax6 = fig.add_subplot(gs[:, 5])
ax4 = fig.add_subplot(gs[0, 4])
ax5 = fig.add_subplot(gs[1, 4])
ax1.set_title("aud_cost_N[-1;-1]")
ax2.set_title("vec_cost_N[-1;-1]")
ax3.set_title("dev. rat.")
ax4.set_title("dist_A")
ax5.set_title("dist_N")
ax6.set_title("Fréchet")
ax7.set_title("Max grad")
#sns.boxplot(y='Naudio_cost_LD', data=df, hue = 'same_speaker', ax = ax1)
#sns.boxplot(y='Ncost_LD', data=df, hue = 'same_speaker', ax = ax2)
#sns.boxplot(y='R', data=df, hue = 'same_speaker', ax = ax3)
#sns.boxplot(y='dist_A', data=df, hue = 'same_speaker', ax = ax4)
#sns.boxplot(y='dist_N', data=df, hue = 'same_speaker', ax = ax5)
#sns.boxplot(y='frechet_dist', data=df, hue = 'same_speaker', ax = ax6)
#sns.boxplot(y='len_devxy', data=df, hue = 'same_speaker', ax = ax7)

sns.boxplot(y='Naudio_cost_LD', data=df, hue = 'same_speaker', ax = ax1)
sns.boxplot(y='Ncost_LD', data=df, hue = 'same_speaker', ax = ax2)
sns.boxplot(y='R', data=df, hue = 'same_speaker', ax = ax3)
sns.boxplot(y='dist_A', data=df, hue = 'same_speaker', ax = ax4)
sns.boxplot(y='dist_N', data=df, hue = 'same_speaker', ax = ax5)
sns.boxplot(y='frechet_dist', data=df, hue = 'same_speaker', ax = ax6)
sns.boxplot(y='Grad_Norm', data=df, hue = 'same_speaker', ax = ax7)
for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
    ax.yaxis.set_label_text("")
fig.figure.savefig(f"{outsuf}_box_{args.layer}.pdf", bbox_inches='tight')
