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
args = parser.parse_args()
from itertools import cycle

def categorize(row,cx,cy):
    if row[cx] == row[cy]:
        return row[cx]  # If L1_x and L1_y are the same, return the language (FR/RU)
    else:
        return str(sorted([row[cx],row[cy]])[0]+" vs "+sorted([row[cx],row[cy]])[1])


lines = ["-","--","-.",":","."]
linecycler = cycle(lines)


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
df1['L1'] = df1.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df2['L1'] = df2.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df3['L1'] = df3.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df4['L1'] = df4.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df5['L1'] = df5.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df6['L1'] = df6.apply(categorize, axis=1, cx='L1_x', cy='L1_y')

# vector
df1_meanR =  df1[df1['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df1_stdR =  df1[df1['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df1_meanF = df1[df1['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df1_stdF = df1[df1['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df1_meanV = df1[df1['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df1_stdV = df1[df1['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df1_meanRref = df1[df1['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m11=df1_meanRref.iloc[1]
df1_meanFref = df1[df1['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m12=df1_meanFref.iloc[1]
df1_meanVref = df1[df1['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m13=df1_meanVref.iloc[1]

# vector
df2_meanR =  df2[df2['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df2_stdR =  df2[df2['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df2_meanF = df2[df2['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df2_stdF = df2[df2['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df2_meanV = df2[df2['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df2_stdV = df2[df2['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df2_meanRref = df2[df2['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m21=df2_meanRref.iloc[1]
df2_meanFref = df2[df2['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m22=df2_meanFref.iloc[1]
df2_meanVref = df2[df2['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m23=df2_meanVref.iloc[1]

df3_meanR =  df3[df3['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df3_stdR =  df3[df3['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df3_meanF = df3[df3['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df3_stdF = df3[df3['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df3_meanV = df3[df3['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df3_stdV = df3[df3['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df3_meanRref = df3[df3['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m31=df3_meanRref.iloc[1]
df3_meanFref = df3[df3['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m32=df3_meanFref.iloc[1]
df3_meanVref = df3[df3['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m33=df3_meanVref.iloc[1]

df4_meanR =  df4[df4['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df4_stdR =  df4[df4['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df4_meanF = df4[df4['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df4_stdF = df4[df4['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df4_meanV = df4[df4['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df4_stdV = df4[df4['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df4_meanRref = df4[df4['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m41=df4_meanRref.iloc[1]
df4_meanFref = df4[df4['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m42=df4_meanFref.iloc[1]
df4_meanVref = df4[df4['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m43=df4_meanVref.iloc[1]

df5_meanR =  df5[df5['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df5_stdR =  df5[df5['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df5_meanF = df5[df5['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df5_stdF = df5[df5['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df5_meanV = df5[df5['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df5_stdV = df5[df5['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df5_meanRref = df5[df5['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m51=df5_meanRref.iloc[1]
df5_meanFref = df5[df5['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m52=df5_meanFref.iloc[1]
df5_meanVref = df5[df5['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m53=df5_meanVref.iloc[1]

df6_meanR =  df6[df6['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df6_stdR =  df6[df6['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df6_meanF = df6[df6['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df6_stdF = df6[df6['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df6_meanV = df6[df6['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df6_stdV = df6[df6['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df6_meanRref = df6[df6['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m61=df6_meanRref.iloc[1]
df6_meanFref = df6[df6['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m62=df6_meanFref.iloc[1]
df6_meanVref = df6[df6['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m63=df6_meanVref.iloc[1]

Lout=[]
for i in range(len(df1_meanR)):
    Lout.append([str(args.input_file1).split("_")[1].split("-")[0], i, m11, m12, m13, df1_meanR.iloc[i],df1_meanF.iloc[i],df1_meanV.iloc[i],(m11-m12)/(m13-m12),(df1_meanR.iloc[i]-df1_meanF.iloc[i])/(df1_meanV.iloc[i]-df1_meanF.iloc[i])])
for i in range(len(df2_meanR)):
    Lout.append([str(args.input_file2).split("_")[1].split("-")[0], i, m21, m22, m23, df2_meanR.iloc[i],df2_meanF.iloc[i],df1_meanV.iloc[i],(m21-m22)/(m23-m22),(df2_meanR.iloc[i]-df2_meanF.iloc[i])/(df2_meanV.iloc[i]-df2_meanF.iloc[i])])
for i in range(len(df3_meanR)):
    Lout.append([str(args.input_file3).split("_")[1].split("-")[0], i, m31, m32, m33, df3_meanR.iloc[i],df3_meanF.iloc[i],df3_meanV.iloc[i],(m31-m32)/(m33-m32),(df3_meanR.iloc[i]-df3_meanF.iloc[i])/(df3_meanV.iloc[i]-df3_meanF.iloc[i])])
for i in range(len(df4_meanR)):
    Lout.append([str(args.input_file4).split("_")[1].split("-")[0], i, m41, m42, m43, df4_meanR.iloc[i],df4_meanF.iloc[i],df4_meanV.iloc[i],(m41-m42)/(m43-m42),(df4_meanR.iloc[i]-df4_meanF.iloc[i])/(df4_meanV.iloc[i]-df4_meanF.iloc[i])])
for i in range(len(df5_meanR)):
    Lout.append([str(args.input_file5).split("_")[1].split("-")[0], i, m51, m52, m53, df5_meanR.iloc[i],df5_meanF.iloc[i],df5_meanV.iloc[i],(m51-m52)/(m53-m52),(df5_meanR.iloc[i]-df5_meanF.iloc[i])/(df5_meanV.iloc[i]-df5_meanF.iloc[i])])
for i in range(len(df6_meanR)):
    Lout.append([str(args.input_file6).split("_")[1].split("-")[0], i, m61, m62, m63, df6_meanR.iloc[i],df6_meanF.iloc[i],df6_meanV.iloc[i],(m61-m62)/(m63-m62),(df6_meanR.iloc[i]-df6_meanF.iloc[i])/(df6_meanV.iloc[i]-df6_meanF.iloc[i])])
dout=pd.DataFrame(Lout)
dout.columns = ["stim","LAY","m1_audio","m2_audio","m3_audio","m1_vect","m2_vect","m3_vect","rat_aud","rat_vect"]

dout.to_csv("audio_v_vect_NORM_thales_pres_L1.csv", index=False)

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
df1['L1'] = df1.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df2['L1'] = df2.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df3['L1'] = df3.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df4['L1'] = df4.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df5['L1'] = df5.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df6['L1'] = df6.apply(categorize, axis=1, cx='L1_x', cy='L1_y')

# vector
df1_meanR =  df1[df1['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df1_stdR =  df1[df1['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df1_meanF = df1[df1['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df1_stdF = df1[df1['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df1_meanV = df1[df1['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df1_stdV = df1[df1['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df1_meanRref = df1[df1['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m11=df1_meanRref.iloc[1]
df1_meanFref = df1[df1['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m12=df1_meanFref.iloc[1]
df1_meanVref = df1[df1['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m13=df1_meanVref.iloc[1]

# vector
df2_meanR =  df2[df2['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df2_stdR =  df2[df2['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df2_meanF = df2[df2['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df2_stdF = df2[df2['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df2_meanV = df2[df2['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df2_stdV = df2[df2['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df2_meanRref = df2[df2['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m21=df2_meanRref.iloc[1]
df2_meanFref = df2[df2['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m22=df2_meanFref.iloc[1]
df2_meanVref = df2[df2['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m23=df2_meanVref.iloc[1]

df3_meanR =  df3[df3['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df3_stdR =  df3[df3['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df3_meanF = df3[df3['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df3_stdF = df3[df3['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df3_meanV = df3[df3['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df3_stdV = df3[df3['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df3_meanRref = df3[df3['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m31=df3_meanRref.iloc[1]
df3_meanFref = df3[df3['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m32=df3_meanFref.iloc[1]
df3_meanVref = df3[df3['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m33=df3_meanVref.iloc[1]

df4_meanR =  df4[df4['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df4_stdR =  df4[df4['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df4_meanF = df4[df4['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df4_stdF = df4[df4['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df4_meanV = df4[df4['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df4_stdV = df4[df4['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df4_meanRref = df4[df4['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m41=df4_meanRref.iloc[1]
df4_meanFref = df4[df4['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m42=df4_meanFref.iloc[1]
df4_meanVref = df4[df4['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m43=df4_meanVref.iloc[1]

df5_meanR =  df5[df5['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df5_stdR =  df5[df5['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df5_meanF = df5[df5['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df5_stdF = df5[df5['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df5_meanV = df5[df5['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df5_stdV = df5[df5['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df5_meanRref = df5[df5['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m51=df5_meanRref.iloc[1]
df5_meanFref = df5[df5['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m52=df5_meanFref.iloc[1]
df5_meanVref = df5[df5['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m53=df5_meanVref.iloc[1]

df6_meanR =  df6[df6['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df6_stdR =  df6[df6['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df6_meanF = df6[df6['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df6_stdF = df6[df6['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df6_meanV = df6[df6['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df6_stdV = df6[df6['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df6_meanRref = df6[df6['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m61=df6_meanRref.iloc[1]
df6_meanFref = df6[df6['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m62=df6_meanFref.iloc[1]
df6_meanVref = df6[df6['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m63=df6_meanVref.iloc[1]

Lout=[]
for i in range(len(df1_meanR)):
    Lout.append([str(args.input_file1).split("_")[1].split("-")[0], i, m11, m12, m13, df1_meanR.iloc[i],df1_meanF.iloc[i],df1_meanV.iloc[i],(m11-m12)/(m13-m12),(df1_meanR.iloc[i]-df1_meanF.iloc[i])/(df1_meanV.iloc[i]-df1_meanF.iloc[i])])
for i in range(len(df2_meanR)):
    Lout.append([str(args.input_file2).split("_")[1].split("-")[0], i, m21, m22, m23, df2_meanR.iloc[i],df2_meanF.iloc[i],df1_meanV.iloc[i],(m21-m22)/(m23-m22),(df2_meanR.iloc[i]-df2_meanF.iloc[i])/(df2_meanV.iloc[i]-df2_meanF.iloc[i])])
for i in range(len(df3_meanR)):
    Lout.append([str(args.input_file3).split("_")[1].split("-")[0], i, m31, m32, m33, df3_meanR.iloc[i],df3_meanF.iloc[i],df3_meanV.iloc[i],(m31-m32)/(m33-m32),(df3_meanR.iloc[i]-df3_meanF.iloc[i])/(df3_meanV.iloc[i]-df3_meanF.iloc[i])])
for i in range(len(df4_meanR)):
    Lout.append([str(args.input_file4).split("_")[1].split("-")[0], i, m41, m42, m43, df4_meanR.iloc[i],df4_meanF.iloc[i],df4_meanV.iloc[i],(m41-m42)/(m43-m42),(df4_meanR.iloc[i]-df4_meanF.iloc[i])/(df4_meanV.iloc[i]-df4_meanF.iloc[i])])
for i in range(len(df5_meanR)):
    Lout.append([str(args.input_file5).split("_")[1].split("-")[0], i, m51, m52, m53, df5_meanR.iloc[i],df5_meanF.iloc[i],df5_meanV.iloc[i],(m51-m52)/(m53-m52),(df5_meanR.iloc[i]-df5_meanF.iloc[i])/(df5_meanV.iloc[i]-df5_meanF.iloc[i])])
for i in range(len(df6_meanR)):
    Lout.append([str(args.input_file6).split("_")[1].split("-")[0], i, m61, m62, m63, df6_meanR.iloc[i],df6_meanF.iloc[i],df6_meanV.iloc[i],(m61-m62)/(m63-m62),(df6_meanR.iloc[i]-df6_meanF.iloc[i])/(df6_meanV.iloc[i]-df6_meanF.iloc[i])])
dout=pd.DataFrame(Lout)
dout.columns = ["stim","LAY","m1_audio","m2_audio","m3_audio","m1_vect","m2_vect","m3_vect","rat_aud","rat_vect"]

dout.to_csv("audio_v_vect_UNNO_thales_pres_L1.csv", index=False)


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
df1['L1'] = df1.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df2['L1'] = df2.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df3['L1'] = df3.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df4['L1'] = df4.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df5['L1'] = df5.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
df6['L1'] = df6.apply(categorize, axis=1, cx='L1_x', cy='L1_y')

# vector
df1_meanR =  df1[df1['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df1_stdR =  df1[df1['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df1_meanF = df1[df1['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df1_stdF = df1[df1['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df1_meanV = df1[df1['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df1_stdV = df1[df1['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df1_meanRref = df1[df1['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m11=df1_meanRref.iloc[1]
df1_meanFref = df1[df1['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m12=df1_meanFref.iloc[1]
df1_meanVref = df1[df1['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m13=df1_meanVref.iloc[1]

# vector
df2_meanR =  df2[df2['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df2_stdR =  df2[df2['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df2_meanF = df2[df2['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df2_stdF = df2[df2['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df2_meanV = df2[df2['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df2_stdV = df2[df2['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df2_meanRref = df2[df2['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m21=df2_meanRref.iloc[1]
df2_meanFref = df2[df2['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m22=df2_meanFref.iloc[1]
df2_meanVref = df2[df2['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m23=df2_meanVref.iloc[1]

df3_meanR =  df3[df3['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df3_stdR =  df3[df3['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df3_meanF = df3[df3['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df3_stdF = df3[df3['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df3_meanV = df3[df3['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df3_stdV = df3[df3['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df3_meanRref = df3[df3['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m31=df3_meanRref.iloc[1]
df3_meanFref = df3[df3['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m32=df3_meanFref.iloc[1]
df3_meanVref = df3[df3['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m33=df3_meanVref.iloc[1]

df4_meanR =  df4[df4['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df4_stdR =  df4[df4['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df4_meanF = df4[df4['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df4_stdF = df4[df4['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df4_meanV = df4[df4['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df4_stdV = df4[df4['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df4_meanRref = df4[df4['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m41=df4_meanRref.iloc[1]
df4_meanFref = df4[df4['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m42=df4_meanFref.iloc[1]
df4_meanVref = df4[df4['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m43=df4_meanVref.iloc[1]

df5_meanR =  df5[df5['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df5_stdR =  df5[df5['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df5_meanF = df5[df5['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df5_stdF = df5[df5['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df5_meanV = df5[df5['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df5_stdV = df5[df5['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df5_meanRref = df5[df5['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m51=df5_meanRref.iloc[1]
df5_meanFref = df5[df5['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m52=df5_meanFref.iloc[1]
df5_meanVref = df5[df5['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m53=df5_meanVref.iloc[1]

df6_meanR =  df6[df6['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
df6_stdR =  df6[df6['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
df6_meanF = df6[df6['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
df6_stdF = df6[df6['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
df6_meanV = df6[df6['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
df6_stdV = df6[df6['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df6_meanRref = df6[df6['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
m61=df6_meanRref.iloc[1]
df6_meanFref = df6[df6['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
m62=df6_meanFref.iloc[1]
df6_meanVref = df6[df6['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
m63=df6_meanVref.iloc[1]

Lout=[]
for i in range(len(df1_meanR)):
    Lout.append([str(args.input_file1).split("_")[1].split("-")[0], i, m11, m12, m13, df1_meanR.iloc[i],df1_meanF.iloc[i],df1_meanV.iloc[i],(m11-m12)/(m13-m12),(df1_meanR.iloc[i]-df1_meanF.iloc[i])/(df1_meanV.iloc[i]-df1_meanF.iloc[i])])
for i in range(len(df2_meanR)):
    Lout.append([str(args.input_file2).split("_")[1].split("-")[0], i, m21, m22, m23, df2_meanR.iloc[i],df2_meanF.iloc[i],df1_meanV.iloc[i],(m21-m22)/(m23-m22),(df2_meanR.iloc[i]-df2_meanF.iloc[i])/(df2_meanV.iloc[i]-df2_meanF.iloc[i])])
for i in range(len(df3_meanR)):
    Lout.append([str(args.input_file3).split("_")[1].split("-")[0], i, m31, m32, m33, df3_meanR.iloc[i],df3_meanF.iloc[i],df3_meanV.iloc[i],(m31-m32)/(m33-m32),(df3_meanR.iloc[i]-df3_meanF.iloc[i])/(df3_meanV.iloc[i]-df3_meanF.iloc[i])])
for i in range(len(df4_meanR)):
    Lout.append([str(args.input_file4).split("_")[1].split("-")[0], i, m41, m42, m43, df4_meanR.iloc[i],df4_meanF.iloc[i],df4_meanV.iloc[i],(m41-m42)/(m43-m42),(df4_meanR.iloc[i]-df4_meanF.iloc[i])/(df4_meanV.iloc[i]-df4_meanF.iloc[i])])
for i in range(len(df5_meanR)):
    Lout.append([str(args.input_file5).split("_")[1].split("-")[0], i, m51, m52, m53, df5_meanR.iloc[i],df5_meanF.iloc[i],df5_meanV.iloc[i],(m51-m52)/(m53-m52),(df5_meanR.iloc[i]-df5_meanF.iloc[i])/(df5_meanV.iloc[i]-df5_meanF.iloc[i])])
for i in range(len(df6_meanR)):
    Lout.append([str(args.input_file6).split("_")[1].split("-")[0], i, m61, m62, m63, df6_meanR.iloc[i],df6_meanF.iloc[i],df6_meanV.iloc[i],(m61-m62)/(m63-m62),(df6_meanR.iloc[i]-df6_meanF.iloc[i])/(df6_meanV.iloc[i]-df6_meanF.iloc[i])])
dout=pd.DataFrame(Lout)
dout.columns = ["stim","LAY","m1_audio","m2_audio","m3_audio","m1_vect","m2_vect","m3_vect","rat_aud","rat_vect"]

dout.to_csv("audio_v_vect_NSPK_thales_pres_L1.csv", index=False)
