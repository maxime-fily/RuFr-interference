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
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", required=True, type=Path)
args = parser.parse_args()

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'

os.chdir("/home/mfily/Documents/NOANOA_locallyowned_texfiles/COLING_2025")
df = pd.read_csv(args.input_file)
mod=str(args.input_file).split("_")[6].replace(".csv","")
for keyword, group in df.groupby("stim"):
    x=group["LAY"]
    y=group["rat_aud"]
    z=group["rat_vect"]
    if len(x) < 25 :
#        print(len(x))
#        print(len(y))
        y = interp1d(x, y, kind="linear", fill_value="extrapolate")(np.arange(0,25))
        z = interp1d(x, z, kind="linear", fill_value="extrapolate")(np.arange(0,25))
        x = np.arange(0,25)
    idx = np.argwhere(np.diff(np.sign(z - y))).flatten()
    plt.figure(figsize=(12, 8))
    print(keyword+" Step 1")
    print(idx)
    if idx.size == 0:
        min_difference=np.min(np.abs(y[:22]-z[:22]))
        min_where=np.argmin(np.abs(y[:22]-z[:22]))
        print(keyword+" Step 2")
        print([min_where])

    # Plot "rat_vect" as a function of "LAY" for the current keyword
    plt.plot(x, y, linestyle="-", label="audio cost ratio")
    plt.plot(x, z, marker="o", linestyle="--", label="vector cost ratio")
    # Customize the plot
    plt.title(f"Evolution of the {mod} modality ratios (stimulus = {keyword})")
    plt.xlabel("layer")
    plt.ylabel("")
    plt.legend()
    plt.grid(True)
    outfile=f"{keyword}_{str(args.input_file).replace('.csv','.pdf')}"
    # Save the plot as a file or show it
    plt.savefig(outfile)  # Saves as PNG file
    plt.clf()

