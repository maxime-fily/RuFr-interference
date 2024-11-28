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
from matplotlib.patches import ConnectionPatch
import scipy.spatial.distance as dist
from scipy.io import wavfile
import librosa
import librosa.display
import textgrid

def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape

    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)

##############################################################################################
# Prog ppal
##############################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", required=True, type=Path)
parser.add_argument("--work_file", required=True, type=Path)

args = parser.parse_args()

df=pd.read_pickle(args.work_dir / args.work_file)

#Filter on target word
df=df[df['words_x'].astype(str) == str(args.work_file).split("-")[0]]

df["filepath_x"]=df["filename_x"]
df["filepath_y"]=df["filename_y"]

df["filegrid_x"] = df["filename_x"].apply(lambda x: textgrid.TextGrid.fromFile(x.with_suffix('.TextGrid')))
df["filegrid_y"] = df["filename_y"].apply(lambda x: textgrid.TextGrid.fromFile(x.with_suffix('.TextGrid')))

#help(df["filegrid_x"][0])

for i, row in df.iterrows():
    if ((row["filegrid_x"][0].name != "words") or (row["filegrid_y"][0].name != "words")):
        print("script is inadequate")
        print(f"{row['filegrid_x'][0].name} is incorrectly named on x (should be 'words', small cap)")
        print(f"{row['filegrid_y'][0].name} is incorrectly named on y (should be 'words', small cap)")
        sys.exit()
    for interval in row["filegrid_x"][0].intervals :
        if interval.mark == row["words_x"]:
            df.at[i,'minTime_x'] = interval.minTime
            df.at[i,'maxTime_x'] = interval.maxTime
    for interval in row["filegrid_y"][0].intervals :
        if interval.mark == row["words_y"]:
            df.at[i,'minTime_y'] = interval.minTime
            df.at[i,'maxTime_y'] = interval.maxTime

#combinations of all speaker_x, speaker_y : by design
df["filename_x"]=df["filename_x"].astype(str).apply(lambda x: x.split("/")[3].split("_")[3].replace(".wav", ""))
df["filename_y"]=df["filename_y"].astype(str).apply(lambda x: x.split("/")[3].split("_")[3].replace(".wav", ""))

#ATTENTION A SUPPRIMER APRES DEV TERMINES !!!!
df = df.sample(n=1)

print(df)
#print(df["interval_x"])
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
#    print(np.max(abs_row))np.array
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
    listR.append([couple['speaker_x'], couple['filename_x'], couple['speaker_y'], couple['filename_y'], a.index1, a.index2, mini, maxi, len(a.index1), (len(a.index1)-mini)/(maxi-mini), listD1, listD2, a1, a2, a.distance, a.normalizedDistance,maxinrow,np.max(abs_row),maxincol,np.max(abs_col),list3m])

    xname=f"{couple['speaker_x']}-{couple['filename_x']}"
    yname=f"{couple['speaker_y']}-{couple['filename_y']}"

#Partie 2 : DTW dans l'espace audio
#X
    f_sx, x = wavfile.read(couple["filepath_x"])
    print("ici")
    print(couple["minTime_x"])
    print(couple["maxTime_x"])
    lower_bound = int(couple["minTime_x"]*f_sx)
    upper_bound = int(couple["maxTime_x"]*f_sx)
    x = x[lower_bound:upper_bound]
    n_fft = int(0.025*f_sx)      # 25 ms
    hop_length = int(0.01*f_sx)  # 10 ms
    mel_spec_x = librosa.feature.melspectrogram(
        y=x/1.0, sr=f_sx, n_mels=40,
        n_fft=n_fft, hop_length=hop_length
        )
    log_mel_spec_x = np.log(mel_spec_x)
    x_seq = log_mel_spec_x.T
    print(x_seq.shape[0])
#Y
    f_sy, y = wavfile.read(couple["filepath_y"])
    print("la")
    print(couple["minTime_y"])
    print(couple["maxTime_y"])
    lower_bound = int(couple["minTime_y"]*f_sy)
    upper_bound = int(couple["maxTime_y"]*f_sy)
    y = y[lower_bound:upper_bound]
    n_fft = int(0.025*f_sy)      # 25 ms
    hop_length = int(0.01*f_sy)  # 10 ms
    mel_spec_y = librosa.feature.melspectrogram(
        y=y/1.0, sr=f_sy, n_mels=40,
        n_fft=n_fft, hop_length=hop_length
        )
    log_mel_spec_y = np.log(mel_spec_y)
    y_seq = log_mel_spec_y.T
    print(y_seq.shape[0])
    dist_mat = dist.cdist(x_seq, y_seq, "cosine")
    path, cost_mat = dp(dist_mat)
    print("Alignment cost: {:.4f}".format(cost_mat[-1, -1]))
#    path=path*49/100
    scaled_path=[]
    for xi, yj in path:
        scaled_path.append((xi*49/100, yj*49/100))
    scaled_path_x, scaled_path_y = zip(*scaled_path)
    scaled_path_x = list(scaled_path_x)
    scaled_path_y = list(scaled_path_y)
    print("en x : ", scaled_path_x)
    print("en y : ", scaled_path_y)
#Partie 3 : plots
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
#    sing_plot = plt.figure(figsize=(3, 2))
#    plt.imshow(b, origin='lower', cmap='viridis', interpolation='nearest')
#    plt.colorbar(label='Cosine Dist')
#    xb = np.arange(b.shape[1])
#    yb = np.arange(b.shape[0])
#    Xb, Yb = np.meshgrid(xb, yb)
#    plt.quiver(Xb, Yb, grad_row, grad_col, color='white', scale=10)
#    plt.plot(maxinrow[1],maxinrow[0], color='red', marker='o', markersize=3, label='')
#    plt.plot(maxincol[1],maxincol[0], color='orange', marker='o', markersize=3, label='')
#    plt.xlabel(yname)
#    plt.ylabel(xname)
#    plt.title('')
#    plt.show()


#3D plot
#    x = np.arange(lcm.T.shape[1])  # 13 values for the columns (axis 1)
#    y = np.arange(lcm.T.shape[0])  # 12 values for the rows (axis 0)
#    A1, A2 = np.meshgrid(x, y)
#    fig = plt.figure()
#    ax = fig.add_subplot(projection = '3d')
#    surf = ax.plot_surface(A1, A2, lcm.T)
#    plt.show()

#Audio DTW plot
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(321)
    ax1.set_xlabel(xname)
    ax1.xaxis.set_label_position('top')
    ax2 = fig.add_subplot(323)
    ax2.set_xlabel(yname)
    ax1.imshow(log_mel_spec_x, origin="lower", interpolation="nearest")
#    ax1.set_xticks(np.arange(0, log_mel_spec_x.shape[1], step=5))
    ax2.imshow(log_mel_spec_y, origin="lower", interpolation="nearest")
    for x_i, y_j in path:
        con = ConnectionPatch(
            xyA=(x_i, 0), xyB=(y_j, log_mel_spec_y.shape[0] - 1), coordsA="data", coordsB="data",
            axesA=ax1, axesB=ax2, color="C7"
            )
        ax2.add_artist(con)
    ax3 = fig.add_subplot(222)
    ax3.imshow(lcm.T, origin='lower', cmap='gray', interpolation='nearest')
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname)
    ax3.plot(a.index1, a.index2, color='red', marker='x', markersize=3, label="vect. DTW")
    ax3.plot(scaled_path_x, scaled_path_y, color='green', marker='o', markersize=3, label="Audio DTW")
    ax3.set_title("Cost Matrix ; alignment cost: {:.4f}".format(cost_mat[-1, -1]))
    xb = np.arange(b.shape[1])
    yb = np.arange(b.shape[0])
    Xb, Yb = np.meshgrid(xb, yb)
    ax4 = fig.add_subplot(224)
    ax4.quiver(Yb, Xb, grad_col, grad_row, color='white', scale=10)
    ax4.plot(maxinrow[0], maxinrow[1],color='red', marker='o', markersize=3, label='')
    ax4.plot(maxincol[0], maxincol[1],color='orange', marker='o', markersize=3, label='')
    ax4.imshow(b.T, origin='lower', cmap='twilight', interpolation='nearest')
    ax4.plot(a.index1, a.index2, color='red', marker='x', markersize=3, label="vect. DTW")
    ax4.plot(scaled_path_x, scaled_path_y, color='green', marker='o', markersize=3, label="Audio DTW")
    ax4.set_title("cosine distance")
    ax4.set_xlabel(xname)
    ax4.set_ylabel(yname)

    plt.show()
    plt.close()

dR = pd.DataFrame(listR)
dR.columns = ["spk_x", "file_x", "spk_y", "file_y", "index1", "index2", "min", "max", "path", "R", 'dev_x', 'dev_y', 'm', 'n', 'dist_A', 'dist_N','MrowGpos','MrowG','McolGpos','McolG','Cdistinfo']
#MrowGpos : maximum row-gradient position ; McolGpos : maximum column-gradient position.
#Cdistinfo : difference between col and row #, list of cos_dist on the diagonal(s), mean ans stdev for this same list.
print("ici")
print(dR.iloc[-1]["index1"])
print("la")
print(dR.iloc[-1]["index2"])



#print("writing to output...")
#output_name = f"table_{str(args.work_file)}"
#dR.to_pickle(args.work_dir / output_name)




