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

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'


UNNO_file1=Path(str(args.input_file1).replace("NORM","UNNO"))
NSPK_file1=Path(str(args.input_file1).replace("NORM","NSPK"))
UNNO_file2=Path(str(args.input_file2).replace("NORM","UNNO"))
NSPK_file2=Path(str(args.input_file2).replace("NORM","NSPK"))
UNNO_file3=Path(str(args.input_file3).replace("NORM","UNNO"))
NSPK_file3=Path(str(args.input_file3).replace("NORM","NSPK"))
UNNO_file4=Path(str(args.input_file4).replace("NORM","UNNO"))
NSPK_file4=Path(str(args.input_file4).replace("NORM","NSPK"))
UNNO_file5=Path(str(args.input_file5).replace("NORM","UNNO"))
NSPK_file5=Path(str(args.input_file5).replace("NORM","NSPK"))
UNNO_file6=Path(str(args.input_file6).replace("NORM","UNNO"))
NSPK_file6=Path(str(args.input_file6).replace("NORM","NSPK"))
os.chdir("/home/mfily/Documents/NOANOA_locallyowned_texfiles/COLING_2025")
df1=pd.read_pickle(args.input_file1)
df2=pd.read_pickle(args.input_file2)
df3=pd.read_pickle(args.input_file3)
df4=pd.read_pickle(args.input_file4)
df5=pd.read_pickle(args.input_file5)
df6=pd.read_pickle(args.input_file6)
df1 = df1[~((df1['spk_x'] == df1['spk_y']) & (df1['file_x'] == df1['file_y']))]
df2 = df2[~((df2['spk_x'] == df2['spk_y']) & (df2['file_x'] == df2['file_y']))]
df3 = df3[~((df3['spk_x'] == df3['spk_y']) & (df3['file_x'] == df3['file_y']))]
df4 = df4[~((df4['spk_x'] == df4['spk_y']) & (df4['file_x'] == df4['file_y']))]
df5 = df5[~((df5['spk_x'] == df5['spk_y']) & (df5['file_x'] == df5['file_y']))]
df6 = df6[~((df6['spk_x'] == df6['spk_y']) & (df6['file_x'] == df6['file_y']))]
df1['same_speaker'] = df1['spk_x'] == df1['spk_y']
df2['same_speaker'] = df2['spk_x'] == df2['spk_y']
df3['same_speaker'] = df3['spk_x'] == df3['spk_y']
df4['same_speaker'] = df4['spk_x'] == df4['spk_y']
df5['same_speaker'] = df5['spk_x'] == df5['spk_y']
df6['same_speaker'] = df6['spk_x'] == df6['spk_y']
dNORM1 = df1[df1['layer'] == args.layer]
dNORM2 = df2[df2['layer'] == args.layer]
dNORM3 = df3[df3['layer'] == args.layer]
dNORM4 = df4[df4['layer'] == args.layer]
dNORM5 = df5[df5['layer'] == args.layer]
dNORM6 = df6[df6['layer'] == args.layer]

df1=pd.read_pickle(UNNO_file1)
df2=pd.read_pickle(UNNO_file2)
df3=pd.read_pickle(UNNO_file3)
df4=pd.read_pickle(UNNO_file4)
df5=pd.read_pickle(UNNO_file5)
df6=pd.read_pickle(UNNO_file6)
df1 = df1[~((df1['spk_x'] == df1['spk_y']) & (df1['file_x'] == df1['file_y']))]
df2 = df2[~((df2['spk_x'] == df2['spk_y']) & (df2['file_x'] == df2['file_y']))]
df3 = df3[~((df3['spk_x'] == df3['spk_y']) & (df3['file_x'] == df3['file_y']))]
df4 = df4[~((df4['spk_x'] == df4['spk_y']) & (df4['file_x'] == df4['file_y']))]
df5 = df5[~((df5['spk_x'] == df5['spk_y']) & (df5['file_x'] == df5['file_y']))]
df6 = df6[~((df6['spk_x'] == df6['spk_y']) & (df6['file_x'] == df6['file_y']))]
df1['same_speaker'] = df1['spk_x'] == df1['spk_y']
df2['same_speaker'] = df2['spk_x'] == df2['spk_y']
df3['same_speaker'] = df3['spk_x'] == df3['spk_y']
df4['same_speaker'] = df4['spk_x'] == df4['spk_y']
df5['same_speaker'] = df5['spk_x'] == df5['spk_y']
df6['same_speaker'] = df6['spk_x'] == df6['spk_y']
dUNNO1 = df1[df1['layer'] == args.layer]
dUNNO2 = df2[df2['layer'] == args.layer]
dUNNO3 = df3[df3['layer'] == args.layer]
dUNNO4 = df4[df4['layer'] == args.layer]
dUNNO5 = df5[df5['layer'] == args.layer]
dUNNO6 = df6[df6['layer'] == args.layer]

df1=pd.read_pickle(NSPK_file1)
df2=pd.read_pickle(NSPK_file2)
df3=pd.read_pickle(NSPK_file3)
df4=pd.read_pickle(NSPK_file4)
df5=pd.read_pickle(NSPK_file5)
df6=pd.read_pickle(NSPK_file6)
df1 = df1[~((df1['spk_x'] == df1['spk_y']) & (df1['file_x'] == df1['file_y']))]
df2 = df2[~((df2['spk_x'] == df2['spk_y']) & (df2['file_x'] == df2['file_y']))]
df3 = df3[~((df3['spk_x'] == df3['spk_y']) & (df3['file_x'] == df3['file_y']))]
df4 = df4[~((df4['spk_x'] == df4['spk_y']) & (df4['file_x'] == df4['file_y']))]
df5 = df5[~((df5['spk_x'] == df5['spk_y']) & (df5['file_x'] == df5['file_y']))]
df6 = df6[~((df6['spk_x'] == df6['spk_y']) & (df6['file_x'] == df6['file_y']))]
df1['same_speaker'] = df1['spk_x'] == df1['spk_y']
df2['same_speaker'] = df2['spk_x'] == df2['spk_y']
df3['same_speaker'] = df3['spk_x'] == df3['spk_y']
df4['same_speaker'] = df4['spk_x'] == df4['spk_y']
df5['same_speaker'] = df5['spk_x'] == df5['spk_y']
df6['same_speaker'] = df6['spk_x'] == df6['spk_y']
dNSPK1 = df1[df1['layer'] == args.layer]
dNSPK2 = df2[df2['layer'] == args.layer]
dNSPK3 = df3[df3['layer'] == args.layer]
dNSPK4 = df4[df4['layer'] == args.layer]
dNSPK5 = df5[df5['layer'] == args.layer]
dNSPK6 = df6[df6['layer'] == args.layer]

dNORM = pd.concat([dNORM1,dNORM2,dNORM3,dNORM4,dNORM5,dNORM6], axis=0)
print(list(dNORM))
dUNNO = pd.concat([dUNNO1,dUNNO2,dUNNO3,dUNNO4,dUNNO5,dUNNO6], axis=0)
dNSPK = pd.concat([dNSPK1,dNSPK2,dNSPK3,dNSPK4,dNSPK5,dNSPK6], axis=0)

fig = plt.figure(figsize=(20, 10))
ax0 = fig.add_subplot(141)
ax1 = fig.add_subplot(142)
ax2 = fig.add_subplot(143)
ax3 = fig.add_subplot(144)
ax0.set_title("Audio")
ax1.set_title("NORM=all")
ax2.set_title("NORM=spk")
ax3.set_title("NORM=no")
sns.violinplot(y='Naudio_cost_LD', data=dNORM, hue = 'same_speaker', ax = ax0)
sns.violinplot(y='Ncost_LD', data=dNORM, hue = 'same_speaker', ax = ax1)
sns.violinplot(y='Ncost_LD', data=dNSPK, hue = 'same_speaker', ax = ax2)
sns.violinplot(y='Ncost_LD', data=dUNNO, hue = 'same_speaker', ax = ax3)
#fig.title("Comparison of three normalization methods (all, per speaker, nil)")
for ax in [ax0, ax1, ax2, ax3]:
    ax.yaxis.set_label_text("")
fig.figure.savefig(f"{outsuf}_vln_compar_Norm_method_{args.layer}.pdf", bbox_inches='tight')
plt.clf()

fig = plt.figure(figsize=(20, 10))
ax0 = fig.add_subplot(141)
ax1 = fig.add_subplot(142)
ax2 = fig.add_subplot(143)
ax3 = fig.add_subplot(144)
ax0.set_title("Audio")
ax1.set_title("NORM=all")
ax2.set_title("NORM=spk")
ax3.set_title("NORM=no")
sns.boxplot(y='Naudio_cost_LD', data=dNORM, hue = 'same_speaker', ax = ax0)
sns.boxplot(y='Ncost_LD', data=dNORM, hue = 'same_speaker', ax = ax1)
sns.boxplot(y='Ncost_LD', data=dNSPK, hue = 'same_speaker', ax = ax2)
sns.boxplot(y='Ncost_LD', data=dUNNO, hue = 'same_speaker', ax = ax3)
#fig.title("Comparison of three normalization methods (all, per speaker, nil)")
for ax in [ax0, ax1, ax2, ax3]:
    ax.yaxis.set_label_text("")
fig.figure.savefig(f"{outsuf}_box_compar_Norm_method_{args.layer}.pdf", bbox_inches='tight')
