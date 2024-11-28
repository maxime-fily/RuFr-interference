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
df1['same_speaker'] = df1['spk_x'] == df1['spk_y']
df2['same_speaker'] = df2['spk_x'] == df2['spk_y']
df3['same_speaker'] = df3['spk_x'] == df3['spk_y']
df4['same_speaker'] = df4['spk_x'] == df4['spk_y']
df5['same_speaker'] = df5['spk_x'] == df5['spk_y']
df6['same_speaker'] = df6['spk_x'] == df6['spk_y']

# vector
df1_meanDS =  df1[df1['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df1_stdDS =  df1[df1['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df1_meanSS = df1[df1['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df1_stdSS = df1[df1['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df1_meanDSref = df1[df1['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m11=df1_meanDSref.iloc[1]
df1_meanSSref = df1[df1['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m12=df1_meanSSref.iloc[1]
df2_meanDS =  df2[df2['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df2_stdDS =  df2[df2['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df2_meanSS = df2[df2['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df2_stdSS = df2[df2['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df2_meanDSref = df2[df2['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m21=df2_meanDSref.iloc[1]
df2_meanSSref = df2[df2['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m22=df2_meanSSref.iloc[1]
df3_meanDS =  df3[df3['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df3_stdDS =  df3[df3['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df3_meanSS = df3[df3['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df3_stdSS = df3[df3['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df3_meanDSref = df3[df3['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m31=df3_meanDSref.iloc[1]
df3_meanSSref = df3[df3['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m32=df3_meanSSref.iloc[1]
df4_meanDS =  df4[df4['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df4_stdDS =  df4[df4['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df4_meanSS = df4[df4['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df4_stdSS = df4[df4['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df4_meanDSref = df4[df4['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m41=df4_meanDSref.iloc[1]
df4_meanSSref = df4[df4['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m42=df4_meanSSref.iloc[1]
df5_meanDS =  df5[df5['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df5_stdDS =  df5[df5['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df5_meanSS = df5[df5['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df5_stdSS = df5[df5['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df5_meanDSref = df5[df5['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m51=df5_meanDSref.iloc[1]
df5_meanSSref = df5[df5['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m52=df5_meanSSref.iloc[1]
df6_meanDS =  df6[df6['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df6_stdDS =  df6[df6['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df6_meanSS = df6[df6['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df6_stdSS = df6[df6['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df6_meanDSref = df6[df6['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m61=df6_meanDSref.iloc[1]
df6_meanSSref = df6[df6['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m62=df6_meanSSref.iloc[1]

Lout=[]
for i in range(len(df1_meanDS)):
    Lout.append([str(args.input_file1).split("_")[1].split("-")[0], i, m11, m12, df1_meanDS.iloc[i],df1_meanSS.iloc[i],m11/m12,df1_meanDS.iloc[i]/df1_meanSS.iloc[i]])
for i in range(len(df2_meanDS)):
    Lout.append([str(args.input_file2).split("_")[1].split("-")[0], i, m21, m22, df2_meanDS.iloc[i],df2_meanSS.iloc[i],m21/m22,df2_meanDS.iloc[i]/df2_meanSS.iloc[i]])
for i in range(len(df3_meanDS)):
    Lout.append([str(args.input_file3).split("_")[1].split("-")[0], i, m31, m32, df3_meanDS.iloc[i],df3_meanSS.iloc[i],m31/m32,df3_meanDS.iloc[i]/df3_meanSS.iloc[i]])
for i in range(len(df4_meanDS)):
    Lout.append([str(args.input_file4).split("_")[1].split("-")[0], i, m41, m42, df4_meanDS.iloc[i],df4_meanSS.iloc[i],m41/m42,df4_meanDS.iloc[i]/df4_meanSS.iloc[i]])
for i in range(len(df5_meanDS)):
    Lout.append([str(args.input_file5).split("_")[1].split("-")[0], i, m51, m52, df5_meanDS.iloc[i],df5_meanSS.iloc[i],m51/m52,df5_meanDS.iloc[i]/df5_meanSS.iloc[i]])
for i in range(len(df6_meanDS)):
    Lout.append([str(args.input_file6).split("_")[1].split("-")[0], i, m61, m62, df6_meanDS.iloc[i],df6_meanSS.iloc[i],m61/m62,df6_meanDS.iloc[i]/df6_meanSS.iloc[i]])
dout=pd.DataFrame(Lout)
dout.columns = ["stim","LAY","m1_audio","m2_audio","m1_vect","m2_vect","rat_aud","rat_vect"]

dout.to_csv("audio_v_vect_NORM_thales_pres.csv", index=False)

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

# vector
df1_meanDS =  df1[df1['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df1_stdDS =  df1[df1['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df1_meanSS = df1[df1['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df1_stdSS = df1[df1['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df1_meanDSref = df1[df1['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m11=df1_meanDSref.iloc[1]
df1_meanSSref = df1[df1['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m12=df1_meanSSref.iloc[1]
df2_meanDS =  df2[df2['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df2_stdDS =  df2[df2['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df2_meanSS = df2[df2['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df2_stdSS = df2[df2['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df2_meanDSref = df2[df2['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m21=df2_meanDSref.iloc[1]
df2_meanSSref = df2[df2['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m22=df2_meanSSref.iloc[1]
df3_meanDS =  df3[df3['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df3_stdDS =  df3[df3['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df3_meanSS = df3[df3['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df3_stdSS = df3[df3['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df3_meanDSref = df3[df3['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m31=df3_meanDSref.iloc[1]
df3_meanSSref = df3[df3['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m32=df3_meanSSref.iloc[1]
df4_meanDS =  df4[df4['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df4_stdDS =  df4[df4['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df4_meanSS = df4[df4['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df4_stdSS = df4[df4['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df4_meanDSref = df4[df4['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m41=df4_meanDSref.iloc[1]
df4_meanSSref = df4[df4['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m42=df4_meanSSref.iloc[1]
df5_meanDS =  df5[df5['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df5_stdDS =  df5[df5['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df5_meanSS = df5[df5['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df5_stdSS = df5[df5['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df5_meanDSref = df5[df5['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m51=df5_meanDSref.iloc[1]
df5_meanSSref = df5[df5['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m52=df5_meanSSref.iloc[1]
df6_meanDS =  df6[df6['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df6_stdDS =  df6[df6['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df6_meanSS = df6[df6['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df6_stdSS = df6[df6['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df6_meanDSref = df6[df6['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m61=df6_meanDSref.iloc[1]
df6_meanSSref = df6[df6['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m62=df6_meanSSref.iloc[1]

Lout=[]
for i in range(len(df1_meanDS)):
    Lout.append([str(args.input_file1).split("_")[1].split("-")[0], i, m11, m12, df1_meanDS.iloc[i],df1_meanSS.iloc[i],m11/m12,df1_meanDS.iloc[i]/df1_meanSS.iloc[i]])
for i in range(len(df2_meanDS)):
    Lout.append([str(args.input_file2).split("_")[1].split("-")[0], i, m21, m22, df2_meanDS.iloc[i],df2_meanSS.iloc[i],m21/m22,df2_meanDS.iloc[i]/df2_meanSS.iloc[i]])
for i in range(len(df3_meanDS)):
    Lout.append([str(args.input_file3).split("_")[1].split("-")[0], i, m31, m32, df3_meanDS.iloc[i],df3_meanSS.iloc[i],m31/m32,df3_meanDS.iloc[i]/df3_meanSS.iloc[i]])
for i in range(len(df4_meanDS)):
    Lout.append([str(args.input_file4).split("_")[1].split("-")[0], i, m41, m42, df4_meanDS.iloc[i],df4_meanSS.iloc[i],m41/m42,df4_meanDS.iloc[i]/df4_meanSS.iloc[i]])
for i in range(len(df5_meanDS)):
    Lout.append([str(args.input_file5).split("_")[1].split("-")[0], i, m51, m52, df5_meanDS.iloc[i],df5_meanSS.iloc[i],m51/m52,df5_meanDS.iloc[i]/df5_meanSS.iloc[i]])
for i in range(len(df6_meanDS)):
    Lout.append([str(args.input_file6).split("_")[1].split("-")[0], i, m61, m62, df6_meanDS.iloc[i],df6_meanSS.iloc[i],m61/m62,df6_meanDS.iloc[i]/df6_meanSS.iloc[i]])
dout=pd.DataFrame(Lout)
dout.columns = ["stim","LAY","m1_audio","m2_audio","m1_vect","m2_vect","rat_aud","rat_vect"]

dout.to_csv("audio_v_vect_UNNO_thales_pres.csv", index=False)


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

# vector
df1_meanDS =  df1[df1['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df1_stdDS =  df1[df1['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df1_meanSS = df1[df1['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df1_stdSS = df1[df1['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df1_meanDSref = df1[df1['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m11=df1_meanDSref.iloc[1]
df1_meanSSref = df1[df1['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m12=df1_meanSSref.iloc[1]
df2_meanDS =  df2[df2['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df2_stdDS =  df2[df2['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df2_meanSS = df2[df2['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df2_stdSS = df2[df2['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df2_meanDSref = df2[df2['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m21=df2_meanDSref.iloc[1]
df2_meanSSref = df2[df2['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m22=df2_meanSSref.iloc[1]
df3_meanDS =  df3[df3['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df3_stdDS =  df3[df3['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df3_meanSS = df3[df3['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df3_stdSS = df3[df3['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df3_meanDSref = df3[df3['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m31=df3_meanDSref.iloc[1]
df3_meanSSref = df3[df3['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m32=df3_meanSSref.iloc[1]
df4_meanDS =  df4[df4['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df4_stdDS =  df4[df4['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df4_meanSS = df4[df4['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df4_stdSS = df4[df4['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df4_meanDSref = df4[df4['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m41=df4_meanDSref.iloc[1]
df4_meanSSref = df4[df4['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m42=df4_meanSSref.iloc[1]
df5_meanDS =  df5[df5['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df5_stdDS =  df5[df5['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df5_meanSS = df5[df5['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df5_stdSS = df5[df5['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df5_meanDSref = df5[df5['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m51=df5_meanDSref.iloc[1]
df5_meanSSref = df5[df5['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m52=df5_meanSSref.iloc[1]
df6_meanDS =  df6[df6['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
df6_stdDS =  df6[df6['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
df6_meanSS = df6[df6['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
df6_stdSS = df6[df6['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# audio cost matrix
df6_meanDSref = df6[df6['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
m61=df6_meanDSref.iloc[1]
df6_meanSSref = df6[df6['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
m62=df6_meanSSref.iloc[1]

Lout=[]
for i in range(len(df1_meanDS)):
    Lout.append([str(args.input_file1).split("_")[1].split("-")[0], i, m11, m12, df1_meanDS.iloc[i],df1_meanSS.iloc[i],m11/m12,df1_meanDS.iloc[i]/df1_meanSS.iloc[i]])
for i in range(len(df2_meanDS)):
    Lout.append([str(args.input_file2).split("_")[1].split("-")[0], i, m21, m22, df2_meanDS.iloc[i],df2_meanSS.iloc[i],m21/m22,df2_meanDS.iloc[i]/df2_meanSS.iloc[i]])
for i in range(len(df3_meanDS)):
    Lout.append([str(args.input_file3).split("_")[1].split("-")[0], i, m31, m32, df3_meanDS.iloc[i],df3_meanSS.iloc[i],m31/m32,df3_meanDS.iloc[i]/df3_meanSS.iloc[i]])
for i in range(len(df4_meanDS)):
    Lout.append([str(args.input_file4).split("_")[1].split("-")[0], i, m41, m42, df4_meanDS.iloc[i],df4_meanSS.iloc[i],m41/m42,df4_meanDS.iloc[i]/df4_meanSS.iloc[i]])
for i in range(len(df5_meanDS)):
    Lout.append([str(args.input_file5).split("_")[1].split("-")[0], i, m51, m52, df5_meanDS.iloc[i],df5_meanSS.iloc[i],m51/m52,df5_meanDS.iloc[i]/df5_meanSS.iloc[i]])
for i in range(len(df6_meanDS)):
    Lout.append([str(args.input_file6).split("_")[1].split("-")[0], i, m61, m62, df6_meanDS.iloc[i],df6_meanSS.iloc[i],m61/m62,df6_meanDS.iloc[i]/df6_meanSS.iloc[i]])
dout=pd.DataFrame(Lout)
dout.columns = ["stim","LAY","m1_audio","m2_audio","m1_vect","m2_vect","rat_aud","rat_vect"]

dout.to_csv("audio_v_vect_NSPK_thales_pres.csv", index=False)
