# this program provides a table line (and a graph, if needed) for all the lines of a table that compares different realizations of a single target word at a time.
# Warning : this program does NOT transpose ANY data, so the reference cost matrix is lcm. A consequence is that X and Y axes have been permuted for graphical represenations.
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

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", required=True, type=Path)
parser.add_argument("--work_file", required=True, type=Path)

args = parser.parse_args()

os.chdir(args.work_dir)
df=pd.read_pickle(args.work_file)

#Filter on target word
df=df[df['words_x'].astype(str) == str(args.work_file).split("-")[0]]
#combinations of all speaker_x, speaker_y : by design
df["filename_x"]=df["filename_x"].astype(str).apply(lambda x: x.split("/")[3].split("_")[3].replace(".wav", ""))
df["filename_y"]=df["filename_y"].astype(str).apply(lambda x: x.split("/")[3].split("_")[3].replace(".wav", ""))

#ATTENTION A SUPPRIMER APRES DEV TERMINES !!!!
df = df.sample(n=1)

#print(list(df))

listR = []

for index, couple in df.iterrows():
#Partie 1 : le chemin optimal
#    print(f"item en cours: {couple}")
    xname=f"{couple['speaker_x']}-{couple['filename_x']}"
    yname=f"{couple['speaker_y']}-{couple['filename_y']}"
#    print(f"En x: {xname}")
#    print(f"En y: {yname}")
    a=couple["DTW"] # DTW
    b=couple["Cdist"]
    a1=a.index1[-1] # indice horizontal
    a2=a.index2[-1] # indice vertical
    lcm=a.costMatrix
    mini=max(a1,a2)
    maxi=a1 + a2
#    print(f"path attendu entre: {mini} et {maxi}")
#    print("pour construire ce path:")
#    print(a.index1)
#    print(a.index2)
    j=0
    k=0
    listD1=[]
    listD2=[]
    for i in range(len(a.index1)-1):
        if (a.index1[i+1]-a.index1[i] == 1):
            listD1.append([a.index1[i],0])
        elif (a.index1[i+1]-a.index1[i] == 0):
            listD1.append([a.index1[i],1])
        else:
            print(a.index1[i+1])
            print(a.index1[i])
            print("erreur")
            sys.exit()

    for i in range(len(a.index2)-1):
        if (a.index2[i+1]-a.index2[i] == 1):
            listD2.append([a.index2[i],0])
        elif (a.index2[i+1]-a.index2[i] == 0):
            listD2.append([a.index2[i],1])
        else:
            print(a.index2[i+1])
            print(a.index2[i])
            print("erreur")
            sys.exit()
#    print(b)
    listcos=[]
    if a1 == a2:
        for i in range(max(a1,a2)):
            listcos.append(b[i,i])
            listcos.append(b[i,i-1])
            listcos.append(b[i-1,i])
    elif a1 > a2:
        for i in range(min(a1,a2)):
            listcos.append(b[i,i])
            for j in range(max(a1,a2)-min(a1,a2)):
                listcos.append(b[i+j,i])
    elif a1 < a2:
        for i in range(min(a1,a2)):
            listcos.append(b[i,i])
            for j in range(max(a1,a2)-min(a1,a2)):
                listcos.append(b[i,i+j])
    list3m=[max(a1,a2)-min(a1,a2),min(listcos),max(listcos),statistics.fmean(listcos),statistics.stdev(listcos)]
    grad_row, grad_col = np.gradient(b)
#    print("row")
#    print(grad_row)
    abs_row=abs(grad_row)
    abs_col=abs(grad_col)
#    print(np.max(abs_row))
#    print(abs_row.argmax())
#    print("col")
#    print(grad_col)
#    print(np.max(abs_col))
#    print(abs_col.argmax())
    maxinrow=unravel_index(abs_row.argmax(), abs_row.shape)
    maxincol=unravel_index(abs_col.argmax(), abs_col.shape)

#    print(maxinrow)
#    print(maxincol)
    if (len(a.index1) != len(a.index2)):
        print("inconsistent length")
        sys.exit()
    listR.append([couple['speaker_x'], couple['filename_x'], couple['speaker_y'], couple['filename_y'], mini, maxi, len(a.index1), (len(a.index1)-mini)/(maxi-mini), listD1, listD2, a1, a2, a.distance, a.normalizedDistance,maxinrow,np.max(abs_row),maxincol,np.max(abs_col),list3m])

    xname=f"{couple['speaker_x']}-{couple['filename_x']}"
    yname=f"{couple['speaker_y']}-{couple['filename_y']}"

# 2D tile plot
#    sing_plot = plt.figure(figsize=(3, 2))
#    plt.imshow(lcm.T, origin='lower', cmap='gray', interpolation='nearest')
#    plt.xlabel(xname)
#    plt.ylabel(yname)
#    plt.title('')
#    plt.show()

# Warning : in the following plot, switching between x and  y is INTENTIONAL.
# => maxinrow[1],maxinrow[0] is in the order, and the xlabel, ylabel are correct.
# Gradient 2D tile plot
    sing_plot = plt.figure(figsize=(3, 2))
    plt.imshow(b, origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cosine Dist')
    xb = np.arange(b.shape[1])
    yb = np.arange(b.shape[0])
    Xb, Yb = np.meshgrid(xb, yb)
    plt.quiver(Xb, Yb, grad_row, grad_col, color='white', scale=10)
    plt.plot(maxinrow[1],maxinrow[0], color='red', marker='o', markersize=3, label='')
    plt.plot(maxincol[1],maxincol[0], color='orange', marker='o', markersize=3, label='')
    plt.xlabel(yname)
    plt.ylabel(xname)
    plt.title('')
    plt.show()


#3D plot
#    x = np.arange(lcm.T.shape[1])  # 13 values for the columns (axis 1)
#    y = np.arange(lcm.T.shape[0])  # 12 values for the rows (axis 0)
#    A1, A2 = np.meshgrid(x, y)
#    fig = plt.figure()
#    ax = fig.add_subplot(projection = '3d')
#    surf = ax.plot_surface(A1, A2, lcm.T)
#    plt.show()

#dR = pd.DataFrame(listR)
#dR.columns = ["spk_x", "file_x", "spk_y", "file_y", "min", "max", "path", "R", 'dev_x', 'dev_y', 'm', 'n', 'dist_A', 'dist_N','MrowGpos','MrowG','McolGpos','McolG','Cdistinfo']
#print(dR)
#print("writing to output...")
#output_name = f"table_{str(args.work_file)}"
#dR.to_pickle(output_name)

#Partie 2 : la matrice de coût
#    lcm=a.costMatrix
#    print(lcm)



