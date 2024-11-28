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
from itertools import cycle
# Input file here is the result*pkl file
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True, type=Path)
args = parser.parse_args()

def categorize(row,cx,cy):
    if row[cx] == row[cy]:
        return row[cx]  # If L1_x and L1_y are the same, return the language (FR/RU)
    else:
        return str(sorted([row[cx],row[cy]])[0]+" vs "+sorted([row[cx],row[cy]])[1])

# graphical settings
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'


lines = ["-","--","-.",":","."]
linecycler = cycle(lines)
outsuf=str(args.input_file).split("-")[0]

UNNO_file=Path(str(args.input_file).replace("NORM","UNNO"))
NSPK_file=Path(str(args.input_file).replace("NORM","NSPK"))
os.chdir("/home/mfily/Documents/NOANOA_locallyowned_texfiles/COLING_2025")
df=pd.read_pickle(args.input_file)
df['same_speaker'] = df['spk_x'] == df['spk_y']
df["L1"] = df.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df['same_gender'] = df['sex_x'] == df['sex_y']
df = df[~((df['spk_x'] == df['spk_y']) & (df['file_x'] == df['file_y']))]
print(list(df))
dNORM = df[['layer','spk_x', 'file_x', 'spk_y', 'file_y', 'same_speaker', 'L1', 'same_gender','Ncost_LD','R','Grad_Norm']]

df=pd.read_pickle(UNNO_file)
df = df[~((df['spk_x'] == df['spk_y']) & (df['file_x'] == df['file_y']))]
df['same_speaker'] = df['spk_x'] == df['spk_y']
df["L1"] = df.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df['same_gender'] = df['sex_x'] == df['sex_y']
dUNNO = df[['layer','spk_x', 'file_x', 'spk_y', 'file_y', 'same_speaker', 'L1', 'same_gender','Ncost_LD','R','Grad_Norm']]

df=pd.read_pickle(NSPK_file)
df['same_speaker'] = df['spk_x'] == df['spk_y']
df["L1"] = df.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df['same_gender'] = df['sex_x'] == df['sex_y']
df = df[~((df['spk_x'] == df['spk_y']) & (df['file_x'] == df['file_y']))]
dNSPK = df[['layer','spk_x', 'file_x', 'spk_y', 'file_y', 'same_speaker', 'L1', 'same_gender','Ncost_LD','R','Grad_Norm']]

fig = plt.figure(figsize=(6, 4))
#ax0 = fig.add_subplot(141)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
plt.subplots_adjust(left=1, bottom=None, right=3, top=None, wspace=None, hspace=None)
sns.lineplot(x='layer', y='R', data=dNORM, hue = 'same_speaker', ax = ax1, palette=['blue', 'orange'], errorbar="sd")
sns.lineplot(x='layer', y='R', data=dNSPK, hue = 'same_speaker', ax = ax1, palette=['green', 'red'],   errorbar="sd")
sns.lineplot(x='layer', y='R', data=dUNNO, hue = 'same_speaker', ax = ax1, palette=['purple', 'cyan'],  errorbar="sd")
sns.lineplot(x='layer', y='Grad_Norm', data=dNORM, hue = 'same_speaker', ax = ax2, palette=['blue', 'orange'],  errorbar="sd")
sns.lineplot(x='layer', y='Grad_Norm', data=dNSPK, hue = 'same_speaker', ax = ax2, palette=['green', 'red'],   errorbar="sd")
sns.lineplot(x='layer', y='Grad_Norm', data=dUNNO, hue = 'same_speaker', ax = ax2, palette=['purple', 'cyan'], errorbar="sd")
sns.lineplot(x='layer', y='Ncost_LD', data=dNORM, hue = 'same_speaker', ax = ax3, palette=['blue', 'orange'], errorbar="sd")
sns.lineplot(x='layer', y='Ncost_LD', data=dNSPK, hue = 'same_speaker', ax = ax3, palette=['green', 'red'], errorbar="sd")
sns.lineplot(x='layer', y='Ncost_LD', data=dUNNO, hue = 'same_speaker', ax = ax3, palette=['purple', 'cyan'], errorbar="sd")
ax1.set_ylabel('dev. rat.',fontsize=16)
ax2.set_ylabel('Grad_Norm',fontsize=16)
ax3.set_ylabel('vec_cost_N',fontsize=16)
ax1.set_xlabel('layer',fontsize=16)
ax2.set_xlabel('layer',fontsize=16)
ax3.set_xlabel('layer',fontsize=16)

# Manually set legends for each subplot
handles1, _ = ax1.get_legend_handles_labels()
handles2, _ = ax2.get_legend_handles_labels()
handles3, _ = ax3.get_legend_handles_labels()
# remove subplot legend
ax1.legend_.remove()
ax2.legend_.remove()
ax3.legend_.remove()

# Add labels for each curve manually
fig.legend(
    handles1[:] ,  # Picking the first 2 lines per plot (ignoring SD)
    ['same speaker, norm=all', 'different speakers, norm=all',
     'same speaker, norm=spk', 'different speakers norm=spk',
     'same speaker, norm=no', 'different speakers norm=no'],
    bbox_to_anchor=(2., -0.2), loc="center", fontsize=16, ncol=3
)
fig.figure.savefig(f"{outsuf}_compar_Norm_method_v_layer.pdf", bbox_inches='tight')

