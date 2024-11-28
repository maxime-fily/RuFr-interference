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
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", required=True, type=Path)
parser.add_argument("--target_word", required=True, type=Path)
parser.add_argument("--normalized", action="store_true")
parser.add_argument("--nspk", action="store_true")
parser.add_argument("--out_dir", required=True, type=Path)
#parser.add_argument("--colhue_x1", required=True, type=str)
#parser.add_argument("--colhue_y1", required=True, type=str)
#parser.add_argument("--colhue_x2", required=True, type=str)
#parser.add_argument("--colhue_y2", required=True, type=str)
args = parser.parse_args()

liste_meta="/home/mfily/Documents/diagnoSTIC_XP/03_make_corpus/03_RUFR/metadata.csv"
listeloc=pd.read_csv(liste_meta, sep="\t")

din=listeloc["ident"]

def categorize(row,cx,cy):
    if row[cx] == row[cy]:
        return row[cx]  # If L1_x and L1_y are the same, return the language (FR/RU)
    else:
        return str(sorted([row[cx],row[cy]])[0]+" vs "+sorted([row[cx],row[cy]])[1])


def elementwise_mean(lists):
    # Stack lists into a 2D NumPy array (rows are lists, columns are elements)
    stacked = np.vstack(lists)

    # Compute the mean for each element across the rows (axis=0)
    mean_list = np.mean(stacked, axis=0)

    return mean_list

def elementwise_stdev(lists):
    # Stack lists into a 2D NumPy array (rows are lists, columns are elements)
    stacked = np.vstack(lists)

    # Compute the mean for each element across the rows (axis=0)
    std_list = np.std(stacked, axis=0)

    return std_list
if args.normalized:
    suffnorm="NORM"
else:
    suffnorm="UNNO"
if args.nspk:
    suffnorm="NSPK"
#Macro DATA
max_len = 0
for l in range(25):
    curr_dir = f"lay_{l}_{args.target_word}_{suffnorm}"
    work_dir = args.root_dir / curr_dir
    work_file = f"table_{args.target_word}-layer_{l}-{suffnorm}.pkl"
    os.chdir(work_dir)
    df=pd.read_pickle(work_file)
    if (df['index1'].apply(len).max() >  max_len):
        max_len = df['index1'].apply(len).max()

print("max length: "+str(max_len))
new_x_mesh = np.linspace(0, 1, max_len)
print("xmesh: ")
print(new_x_mesh)
listI=[]
listH=[]
listH8=[]
listA=[]
c_it=['AB2', 'KL', 'KV', 'MD', 'AN', 'SD', 'YC', 'UV']
plt.set_cmap('tab20')

#graphical settings
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'

lines = ["-","--","-.",":","."]
linecycler = cycle(lines)

if __name__ == "__main__":
#Initialize empty dataframe for data aggregation
    dg=pd.DataFrame()
    for i in range(25):
        curr_dir = f"lay_{i}_{args.target_word}_{suffnorm}"
        work_dir = args.root_dir / curr_dir
        work_file = f"table_{args.target_word}-layer_{i}-{suffnorm}.pkl"
        os.chdir(work_dir)
        df=pd.read_pickle(work_file)

#add the other indicators derived from objects present in the dataframe
        listCOMPL=[]
        for index, couple in df.iterrows():
            #COST MATRIX
            last_digit=couple["DTW"].costMatrix[-1, -1]
            audio_last_digit=couple["audio_cost_mat"][-1, -1]
            #SCATTER INDICES
            dev_x_indices = [x for x, y in couple['dev_x'] if y != 0]
            first_component_dev_y = [couple['dev_y'][index][0] for index, (x, y) in enumerate(couple['dev_x']) if y != 0]
            zip_horiz = zip(dev_x_indices, first_component_dev_y)
            dev_y_indices = [x for x, y in couple['dev_y'] if y != 0]
            first_component_dev_x = [couple['dev_x'][index][0] for index, (x, y) in enumerate(couple['dev_y']) if y != 0]
            zip_vertic = zip(dev_y_indices, first_component_dev_x)
            zipT = list(zip_horiz) + list(zip_vertic)
            #ANCIENNE VERSION
            #AX, AY = zip(*couple['audio_DTW_path'])
            #NOUVELLE VERSION
            AX, AY = zip(*couple['scaled_audio_path'])
            AX, AY = np.array(AX), np.array(AY)
            VPATH = list(zip(couple['index1'],couple['index2']))
            vect_x = np.array([AX for AX, _ in VPATH])
            interp_y = np.interp(vect_x, AX, AY)
            interpolated_audio_path = list(zip(vect_x, interp_y))
            frechet=frdist(interpolated_audio_path,VPATH)
            assert len(VPATH) == len(interpolated_audio_path)
            #Euclidean
            distances = []
            #Manhattan
            distancesM = []
            for j in range(len(VPATH)):
                x1, y1 = VPATH[j]
                x2, y2 = interpolated_audio_path[j]
                dist = float("{:.2f}".format(np.sqrt((x2 - x1)**2 + (y2 - y1)**2)))
                distM = np.abs(x2 - x1) + np.abs(y2 - y1)
                distances.append(dist)
                distancesM.append(distM)

    #interpolate the point_dist values
#            print(distances)
#ANCIENNE VERSION
            #listCOMPL.append([couple['DTW'], last_digit, last_digit/(couple['m']*couple['n']), couple['spk_x'], couple['file_x'], couple['spk_y'], couple['file_y'], couple['index1'], couple['index2'], couple['min'], couple['max'], couple['vect_path'], \
                #couple['R'], couple['dev_x'], couple['dev_y'], zipT, len(zipT), couple['m'], couple['n'], couple['dist_A'], couple['dist_N'], couple['MrowGpos'], couple['MrowG'],  couple['McolGpos'], couple['McolG'],  couple['Cdistinfo'], \
                #interpolated_audio_path, couple['audio_DTW_path'],  couple['audio_path'],  couple['audio_cost_mat'],  audio_last_digit, audio_last_digit/(couple["audio_cost_mat"].shape[0]*couple["audio_cost_mat"].shape[1]), frechet, distances, distancesM])
#NOUVELLE VERSION
            listCOMPL.append([couple['DTW'], last_digit, last_digit/(couple['m']+couple['n']), couple['spk_x'], couple['file_x'], couple['spk_y'], couple['file_y'], couple['index1'], couple['index2'], couple['min'], couple['max'], couple['vect_path_len'], \
                couple['R'], couple['dev_x'], couple['dev_y'], zipT, len(zipT), couple['m'], couple['n'], couple['dist_A'], couple['dist_N'], couple['MrowGpos'], couple['MrowG'],  couple['McolGpos'], couple['McolG'],  couple['Cdistinfo'], \
                interpolated_audio_path, couple['scaled_audio_path'],  couple['audio_path'],  couple['audio_cost_mat'],  audio_last_digit, audio_last_digit/(couple['m']+couple['n']), frechet, distances, distancesM])
        df = pd.DataFrame(listCOMPL)
        df.columns = ["DTW","cost_LD","Ncost_LD","spk_x", "file_x", "spk_y", "file_y", "index1", "index2", "min", "max", "vect_path_len", "R", 'dev_x', 'dev_y', 'dev_coordxy', 'len_devxy', 'm', 'n', 'dist_A', 'dist_N','MrowGpos','MrowG','McolGpos','McolG','Cdistinfo','interpolated_audio_path','audio_scaled_path',"audio_path", "audio_cost_mat", "audio_cost_LD", "Naudio_cost_LD","frechet_dist","point_dist","manh_dist"]
        interpol_y=[]
        manhat_y=[]
#filter dataframe so that it does not count the same-same cos dist., couple[''], couple[''],
#filter dataframe so that it does not count the same-same cos dist.
#This is important to avoid a bias towards the low values of distance, and it has no interest to keep them anyway.
        df = df[~((df['spk_x'] == df['spk_y']) & (df['file_x'] == df['file_y']))]
#set all durations to 1
        for idx, cpl in df.iterrows():
            original_x_mesh = np.arange(len(cpl["point_dist"])) / len(cpl["point_dist"])  # Original x-values (between 0 and 1)
            interpo_y = np.interp(new_x_mesh, original_x_mesh, cpl["point_dist"])
            manh_y = np.interp(new_x_mesh, original_x_mesh, cpl["manh_dist"])
            interpol_y.append(interpo_y)
            manhat_y.append(manh_y)
        print(len(interpol_y))

        merged_df = pd.merge(df, listeloc[['ident', 'L1', 'age', 'sex', 'lvl_fr' ,'num_lev_fr', 'lvl_ru' ,'num_lev_ru']], left_on='spk_x', right_on='ident', how='left')
        merged_df = merged_df.rename(columns={'L1': 'L1_x'})
        merged_df = merged_df.rename(columns={'age': 'age_x'})
        merged_df = merged_df.rename(columns={'sex': 'sex_x'})
        merged_df = merged_df.rename(columns={'lvl_fr': 'level_fr_x'})
        merged_df = merged_df.rename(columns={'num_lev_fr': 'num_lev_fr_x'})
        merged_df = merged_df.rename(columns={'lvl_ru': 'level_ru_x'})
        merged_df = merged_df.rename(columns={'num_lev_ru': 'num_lev_ru_x'})
        merged_df = merged_df.drop(columns=['ident'])
        db_merged = pd.merge(merged_df, listeloc[['ident', 'L1', 'age', 'sex', 'lvl_fr' ,'num_lev_fr', 'lvl_ru' ,'num_lev_ru']], left_on='spk_y', right_on='ident', how='left')
        db_merged = db_merged.rename(columns={'L1': 'L1_y'})
        db_merged = db_merged.rename(columns={'age': 'age_y'})
        db_merged = db_merged.rename(columns={'sex': 'sex_y'})
        db_merged = db_merged.rename(columns={'lvl_fr': 'level_fr_y'})
        db_merged = db_merged.rename(columns={'num_lev_fr': 'num_lev_fr_y'})
        db_merged = db_merged.rename(columns={'lvl_ru': 'level_ru_y'})
        db_merged = db_merged.rename(columns={'num_lev_ru': 'num_lev_ru_y'})
        db_merged = db_merged.drop(columns=['ident'])
        #print(len(df))
        #print(len(merged_df))
        #print(len(db_merged))
        #print(db_merged)
        #Ajout d'un label same speaker vs different speaker
        db_merged['same_speaker'] = db_merged['spk_x'] == db_merged['spk_y']
        db_merged["L1"] = db_merged.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
        db_merged['same_gender'] = db_merged['sex_x'] == db_merged['sex_y']
        db_merged['layer'] = i

#########################################################
#INDIV SPK case
#########################################################

#mean values
        dA_0 = pd.DataFrame({"interp_point_dist": pd.Series(interpol_y)})
        dA=pd.concat([db_merged[['spk_x', 'spk_y','L1_x', 'L1_y', 'same_speaker']],dA_0], axis=1)
        dA0=dA[((dA['spk_x'] == c_it[0])&(dA['spk_y'] == c_it[0]))]
        dA1=dA[((dA['spk_x'] == c_it[1])&(dA['spk_y'] == c_it[1]))]
        dA2=dA[((dA['spk_x'] == c_it[2])&(dA['spk_y'] == c_it[2]))]
        dA3=dA[((dA['spk_x'] == c_it[3])&(dA['spk_y'] == c_it[3]))]
        dA4=dA[((dA['spk_x'] == c_it[4])&(dA['spk_y'] == c_it[4]))]
        dA5=dA[((dA['spk_x'] == c_it[5])&(dA['spk_y'] == c_it[5]))]
        dA6=dA[((dA['spk_x'] == c_it[6])&(dA['spk_y'] == c_it[6]))]
        dA7=dA[((dA['spk_x'] == c_it[7])&(dA['spk_y'] == c_it[7]))]
        dA0_stack=np.vstack(dA0['interp_point_dist'])
        dA0_means = np.mean(dA0_stack, axis=0)
        dA0_std = np.std(dA0_stack, axis=0)
        dA1_stack=np.vstack(dA1['interp_point_dist'])
        dA1_means = np.mean(dA1_stack, axis=0)
        dA1_std = np.std(dA1_stack, axis=0)
        dA2_stack=np.vstack(dA2['interp_point_dist'])
        dA2_means = np.mean(dA2_stack, axis=0)
        dA2_std = np.std(dA2_stack, axis=0)
        dA3_stack=np.vstack(dA3['interp_point_dist'])
        dA3_means = np.mean(dA3_stack, axis=0)
        dA3_std = np.std(dA3_stack, axis=0)
        dA4_stack=np.vstack(dA4['interp_point_dist'])
        dA4_means = np.mean(dA4_stack, axis=0)
        dA4_std = np.std(dA4_stack, axis=0)
        dA5_stack=np.vstack(dA5['interp_point_dist'])
        dA5_means = np.mean(dA5_stack, axis=0)
        dA5_std = np.std(dA5_stack, axis=0)
        dA6_stack=np.vstack(dA6['interp_point_dist'])
        dA6_means = np.mean(dA6_stack, axis=0)
        dA6_std = np.std(dA6_stack, axis=0)
        dA7_stack=np.vstack(dA7['interp_point_dist'])
        dA7_means = np.mean(dA7_stack, axis=0)
        dA7_std = np.std(dA7_stack, axis=0)

        listA.append([i, dA0_means, dA0_std, dA1_means, dA1_std, dA2_means, dA2_std, dA3_means, dA3_std, dA4_means, dA4_std, dA5_means, dA5_std, dA6_means, dA6_std, dA7_means, dA7_std])
    DA=pd.DataFrame(listA, columns=["LAY", f"mean_point_dist_{c_it[0]}", f"std_point_dist_{c_it[0]}", f"mean_point_dist_{c_it[1]}", f"std_point_dist_{c_it[1]}", f"mean_point_dist_{c_it[2]}", f"std_point_dist_{c_it[2]}", f"mean_point_dist_{c_it[3]}", f"std_point_dist_{c_it[3]}", f"mean_point_dist_{c_it[4]}", f"std_point_dist_{c_it[4]}", f"mean_point_dist_{c_it[5]}", f"std_point_dist_{c_it[5]}", f"mean_point_dist_{c_it[6]}", f"std_point_dist_{c_it[6]}", f"mean_point_dist_{c_it[7]}", f"std_point_dist_{c_it[7]}"])
    print(list(DA))
    os.chdir(args.out_dir)
    DA.to_pickle("out_for_tulle.pkl")

#INDIV SPK
    fig = plt.figure()
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    x_value=new_x_mesh
    y_value1=DA[f"mean_point_dist_{c_it[0]}"].iloc[16]
    z_value1=DA[f"std_point_dist_{c_it[0]}"].iloc[16]
    y_value2=DA[f"mean_point_dist_{c_it[1]}"].iloc[16]
    z_value2=DA[f"std_point_dist_{c_it[1]}"].iloc[16]
    y_value3=DA[f"mean_point_dist_{c_it[2]}"].iloc[16]
    z_value3=DA[f"std_point_dist_{c_it[2]}"].iloc[16]
    y_value4=DA[f"mean_point_dist_{c_it[3]}"].iloc[16]
    z_value4=DA[f"std_point_dist_{c_it[3]}"].iloc[16]
    y_value5=DA[f"mean_point_dist_{c_it[4]}"].iloc[16]
    z_value5=DA[f"std_point_dist_{c_it[4]}"].iloc[16]
    y_value6=DA[f"mean_point_dist_{c_it[5]}"].iloc[16]
    z_value6=DA[f"std_point_dist_{c_it[5]}"].iloc[16]
    y_value7=DA[f"mean_point_dist_{c_it[6]}"].iloc[16]
    z_value7=DA[f"std_point_dist_{c_it[6]}"].iloc[16]
    y_value8=DA[f"mean_point_dist_{c_it[7]}"].iloc[16]
    z_value8=DA[f"std_point_dist_{c_it[7]}"].iloc[16]
    plt.plot(x_value,y_value1, next(linecycler),linewidth=0.5,label=f"{c_it[0]}_{listeloc.loc[listeloc['ident'] == c_it[0],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[0],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value1 - z_value1, y_value1 + z_value1, alpha=0.1)
    plt.plot(x_value,y_value2, next(linecycler),linewidth=0.5,label=f"{c_it[1]}_{listeloc.loc[listeloc['ident'] == c_it[1],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[1],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value2 - z_value2, y_value2 + z_value2, alpha=0.1)
    plt.plot(x_value,y_value3, next(linecycler),linewidth=0.5,label=f"{c_it[2]}_{listeloc.loc[listeloc['ident'] == c_it[2],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[2],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value3 - z_value3, y_value3 + z_value3, alpha=0.1)
    plt.plot(x_value,y_value4, next(linecycler),linewidth=0.5,label=f"{c_it[3]}_{listeloc.loc[listeloc['ident'] == c_it[3],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[3],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value4 - z_value4, y_value4 + z_value4, alpha=0.1)
    plt.plot(x_value,y_value5, next(linecycler),linewidth=0.5,label=f"{c_it[4]}_{listeloc.loc[listeloc['ident'] == c_it[4],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[4],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value5 - z_value5, y_value5 + z_value5, alpha=0.1)
    plt.plot(x_value,y_value6, next(linecycler),linewidth=0.5,label=f"{c_it[5]}_{listeloc.loc[listeloc['ident'] == c_it[5],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[5],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value6 - z_value6, y_value6 + z_value6, alpha=0.1)
    plt.plot(x_value,y_value7, next(linecycler),linewidth=0.5,label=f"{c_it[6]}_{listeloc.loc[listeloc['ident'] == c_it[6],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[6],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value7 - z_value7, y_value7 + z_value7, alpha=0.1)
    plt.plot(x_value,y_value8, next(linecycler),linewidth=0.5,label=f"{c_it[7]}_{listeloc.loc[listeloc['ident'] == c_it[7],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[7],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value8 - z_value8, y_value8 + z_value8, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.02, 0.7), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_eff_indiv_L16_for_tulle.pdf", bbox_inches='tight')
    plt.legend()
    plt.close()

    fig = plt.figure()
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    x_value=new_x_mesh
    y_value1=DA[f"mean_point_dist_{c_it[0]}"].iloc[19]
    z_value1=DA[f"std_point_dist_{c_it[0]}"].iloc[19]
    y_value2=DA[f"mean_point_dist_{c_it[1]}"].iloc[19]
    z_value2=DA[f"std_point_dist_{c_it[1]}"].iloc[19]
    y_value3=DA[f"mean_point_dist_{c_it[2]}"].iloc[19]
    z_value3=DA[f"std_point_dist_{c_it[2]}"].iloc[19]
    y_value4=DA[f"mean_point_dist_{c_it[3]}"].iloc[19]
    z_value4=DA[f"std_point_dist_{c_it[3]}"].iloc[19]
    y_value5=DA[f"mean_point_dist_{c_it[4]}"].iloc[19]
    z_value5=DA[f"std_point_dist_{c_it[4]}"].iloc[19]
    y_value6=DA[f"mean_point_dist_{c_it[5]}"].iloc[19]
    z_value6=DA[f"std_point_dist_{c_it[5]}"].iloc[19]
    y_value7=DA[f"mean_point_dist_{c_it[6]}"].iloc[19]
    z_value7=DA[f"std_point_dist_{c_it[6]}"].iloc[19]
    y_value8=DA[f"mean_point_dist_{c_it[7]}"].iloc[19]
    z_value8=DA[f"std_point_dist_{c_it[7]}"].iloc[19]
    plt.plot(x_value,y_value1, next(linecycler),linewidth=0.5,label=f"{c_it[0]}_{listeloc.loc[listeloc['ident'] == c_it[0],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[0],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value1 - z_value1, y_value1 + z_value1, alpha=0.1)
    plt.plot(x_value,y_value2, next(linecycler),linewidth=0.5,label=f"{c_it[1]}_{listeloc.loc[listeloc['ident'] == c_it[1],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[1],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value2 - z_value2, y_value2 + z_value2, alpha=0.1)
    plt.plot(x_value,y_value3, next(linecycler),linewidth=0.5,label=f"{c_it[2]}_{listeloc.loc[listeloc['ident'] == c_it[2],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[2],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value3 - z_value3, y_value3 + z_value3, alpha=0.1)
    plt.plot(x_value,y_value4, next(linecycler),linewidth=0.5,label=f"{c_it[3]}_{listeloc.loc[listeloc['ident'] == c_it[3],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[3],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value4 - z_value4, y_value4 + z_value4, alpha=0.1)
    plt.plot(x_value,y_value5, next(linecycler),linewidth=0.5,label=f"{c_it[4]}_{listeloc.loc[listeloc['ident'] == c_it[4],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[4],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value5 - z_value5, y_value5 + z_value5, alpha=0.1)
    plt.plot(x_value,y_value6, next(linecycler),linewidth=0.5,label=f"{c_it[5]}_{listeloc.loc[listeloc['ident'] == c_it[5],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[5],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value6 - z_value6, y_value6 + z_value6, alpha=0.1)
    plt.plot(x_value,y_value7, next(linecycler),linewidth=0.5,label=f"{c_it[6]}_{listeloc.loc[listeloc['ident'] == c_it[6],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[6],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value7 - z_value7, y_value7 + z_value7, alpha=0.1)
    plt.plot(x_value,y_value8, next(linecycler),linewidth=0.5,label=f"{c_it[7]}_{listeloc.loc[listeloc['ident'] == c_it[7],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[7],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value8 - z_value8, y_value8 + z_value8, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.02, 0.7), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_eff_indiv_L19_for_tulle.pdf", bbox_inches='tight')
    plt.legend()
    plt.close()

    fig = plt.figure()
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    x_value=new_x_mesh
    y_value1=DA[f"mean_point_dist_{c_it[0]}"].iloc[20]
    z_value1=DA[f"std_point_dist_{c_it[0]}"].iloc[20]
    y_value2=DA[f"mean_point_dist_{c_it[1]}"].iloc[20]
    z_value2=DA[f"std_point_dist_{c_it[1]}"].iloc[20]
    y_value3=DA[f"mean_point_dist_{c_it[2]}"].iloc[20]
    z_value3=DA[f"std_point_dist_{c_it[2]}"].iloc[20]
    y_value4=DA[f"mean_point_dist_{c_it[3]}"].iloc[20]
    z_value4=DA[f"std_point_dist_{c_it[3]}"].iloc[20]
    y_value5=DA[f"mean_point_dist_{c_it[4]}"].iloc[20]
    z_value5=DA[f"std_point_dist_{c_it[4]}"].iloc[20]
    y_value6=DA[f"mean_point_dist_{c_it[5]}"].iloc[20]
    z_value6=DA[f"std_point_dist_{c_it[5]}"].iloc[20]
    y_value7=DA[f"mean_point_dist_{c_it[6]}"].iloc[20]
    z_value7=DA[f"std_point_dist_{c_it[6]}"].iloc[20]
    y_value8=DA[f"mean_point_dist_{c_it[7]}"].iloc[20]
    z_value8=DA[f"std_point_dist_{c_it[7]}"].iloc[20]
    plt.plot(x_value,y_value1, next(linecycler),linewidth=0.5,label=f"{c_it[0]}_{listeloc.loc[listeloc['ident'] == c_it[0],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[0],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value1 - z_value1, y_value1 + z_value1, alpha=0.1)
    plt.plot(x_value,y_value2, next(linecycler),linewidth=0.5,label=f"{c_it[1]}_{listeloc.loc[listeloc['ident'] == c_it[1],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[1],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value2 - z_value2, y_value2 + z_value2, alpha=0.1)
    plt.plot(x_value,y_value3, next(linecycler),linewidth=0.5,label=f"{c_it[2]}_{listeloc.loc[listeloc['ident'] == c_it[2],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[2],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value3 - z_value3, y_value3 + z_value3, alpha=0.1)
    plt.plot(x_value,y_value4, next(linecycler),linewidth=0.5,label=f"{c_it[3]}_{listeloc.loc[listeloc['ident'] == c_it[3],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[3],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value4 - z_value4, y_value4 + z_value4, alpha=0.1)
    plt.plot(x_value,y_value5, next(linecycler),linewidth=0.5,label=f"{c_it[4]}_{listeloc.loc[listeloc['ident'] == c_it[4],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[4],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value5 - z_value5, y_value5 + z_value5, alpha=0.1)
    plt.plot(x_value,y_value6, next(linecycler),linewidth=0.5,label=f"{c_it[5]}_{listeloc.loc[listeloc['ident'] == c_it[5],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[5],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value6 - z_value6, y_value6 + z_value6, alpha=0.1)
    plt.plot(x_value,y_value7, next(linecycler),linewidth=0.5,label=f"{c_it[6]}_{listeloc.loc[listeloc['ident'] == c_it[6],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[6],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value7 - z_value7, y_value7 + z_value7, alpha=0.1)
    plt.plot(x_value,y_value8, next(linecycler),linewidth=0.5,label=f"{c_it[7]}_{listeloc.loc[listeloc['ident'] == c_it[7],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[7],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value8 - z_value8, y_value8 + z_value8, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.02, 0.7), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_eff_indiv_L20_for_tulle.pdf", bbox_inches='tight')
    plt.legend()
    plt.close()

    fig = plt.figure()
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    x_value=new_x_mesh
    y_value1=DA[f"mean_point_dist_{c_it[0]}"].iloc[5]
    z_value1=DA[f"std_point_dist_{c_it[0]}"].iloc[5]
    y_value2=DA[f"mean_point_dist_{c_it[1]}"].iloc[5]
    z_value2=DA[f"std_point_dist_{c_it[1]}"].iloc[5]
    y_value3=DA[f"mean_point_dist_{c_it[2]}"].iloc[5]
    z_value3=DA[f"std_point_dist_{c_it[2]}"].iloc[5]
    y_value4=DA[f"mean_point_dist_{c_it[3]}"].iloc[5]
    z_value4=DA[f"std_point_dist_{c_it[3]}"].iloc[5]
    y_value5=DA[f"mean_point_dist_{c_it[4]}"].iloc[5]
    z_value5=DA[f"std_point_dist_{c_it[4]}"].iloc[5]
    y_value6=DA[f"mean_point_dist_{c_it[5]}"].iloc[5]
    z_value6=DA[f"std_point_dist_{c_it[5]}"].iloc[5]
    y_value7=DA[f"mean_point_dist_{c_it[6]}"].iloc[5]
    z_value7=DA[f"std_point_dist_{c_it[6]}"].iloc[5]
    y_value8=DA[f"mean_point_dist_{c_it[7]}"].iloc[5]
    z_value8=DA[f"std_point_dist_{c_it[7]}"].iloc[5]
    plt.plot(x_value,y_value1, next(linecycler),linewidth=0.5,label=f"{c_it[0]}_{listeloc.loc[listeloc['ident'] == c_it[0],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[0],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value1 - z_value1, y_value1 + z_value1, alpha=0.1)
    plt.plot(x_value,y_value2, next(linecycler),linewidth=0.5,label=f"{c_it[1]}_{listeloc.loc[listeloc['ident'] == c_it[1],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[1],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value2 - z_value2, y_value2 + z_value2, alpha=0.1)
    plt.plot(x_value,y_value3, next(linecycler),linewidth=0.5,label=f"{c_it[2]}_{listeloc.loc[listeloc['ident'] == c_it[2],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[2],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value3 - z_value3, y_value3 + z_value3, alpha=0.1)
    plt.plot(x_value,y_value4, next(linecycler),linewidth=0.5,label=f"{c_it[3]}_{listeloc.loc[listeloc['ident'] == c_it[3],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[3],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value4 - z_value4, y_value4 + z_value4, alpha=0.1)
    plt.plot(x_value,y_value5, next(linecycler),linewidth=0.5,label=f"{c_it[4]}_{listeloc.loc[listeloc['ident'] == c_it[4],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[4],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value5 - z_value5, y_value5 + z_value5, alpha=0.1)
    plt.plot(x_value,y_value6, next(linecycler),linewidth=0.5,label=f"{c_it[5]}_{listeloc.loc[listeloc['ident'] == c_it[5],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[5],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value6 - z_value6, y_value6 + z_value6, alpha=0.1)
    plt.plot(x_value,y_value7, next(linecycler),linewidth=0.5,label=f"{c_it[6]}_{listeloc.loc[listeloc['ident'] == c_it[6],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[6],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value7 - z_value7, y_value7 + z_value7, alpha=0.1)
    plt.plot(x_value,y_value8, next(linecycler),linewidth=0.5,label=f"{c_it[7]}_{listeloc.loc[listeloc['ident'] == c_it[7],'L1'].iloc[0]}_{listeloc.loc[listeloc['ident'] == c_it[7],'sex'].iloc[0]}")
    plt.fill_between(x_value, y_value8 - z_value8, y_value8 + z_value8, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.02, 0.7), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_eff_indiv_L5_for_tulle.pdf", bbox_inches='tight')
    plt.legend()
    plt.close()











