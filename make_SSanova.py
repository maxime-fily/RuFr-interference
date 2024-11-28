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
c_it=['KL', 'KV', 'MD', 'AN', 'SD', 'YC', 'UV', 'RN']
plt.set_cmap('tab20')

#graphical settings
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'

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
#19SPK
#########################################################
#mean values : eucl
        dh0 = pd.DataFrame({"interp_point_dist": pd.Series(interpol_y)})
        dh=pd.concat([db_merged[['spk_x', 'spk_y','L1_x', 'L1_y', 'same_speaker']],dh0], axis=1)
        dh['same_L1'] =  dh['L1_x'] == dh['L1_y']
        dh_stack=np.vstack(dh['interp_point_dist'])
        dh_means = np.mean(dh_stack, axis=0)
        dh_std = np.std(dh_stack, axis=0)
#mean values : manhattan
        di0 = pd.DataFrame({"manhat_point_dist": pd.Series(manhat_y)})
        di=pd.concat([db_merged[['spk_x', 'spk_y','L1_x', 'L1_y', 'same_speaker']],di0], axis=1)
        di['same_L1'] =  di['L1_x'] == di['L1_y']
        di_stack=np.vstack(di['manhat_point_dist'])
        di_means = np.mean(di_stack, axis=0)
        di_std = np.std(di_stack, axis=0)
#spk
        dh_stackSS=np.vstack(dh[dh['same_speaker']]['interp_point_dist'])
        dh_meansSS = np.mean(dh_stackSS, axis=0)
        dh_stdSS = np.std(dh_stackSS, axis=0)
        dh_stackDS=np.vstack(dh[~dh['same_speaker']]['interp_point_dist'])
        dh_meansDS = np.mean(dh_stackDS, axis=0)
        dh_stdDS = np.std(dh_stackDS, axis=0)
#L1
        dh_stackSL1=np.vstack(dh[dh['same_L1']]['interp_point_dist'])
        dh_meansSL1 = np.mean(dh_stackSL1, axis=0)
        dh_stdSL1 = np.std(dh_stackSL1, axis=0)
        dh_stackDL1=np.vstack(dh[~dh['same_L1']]['interp_point_dist'])
        dh_meansDL1 = np.mean(dh_stackDL1, axis=0)
        dh_stdDL1 = np.std(dh_stackDL1, axis=0)

        listH.append([i, dh_means, dh_std, dh_meansSS, dh_stdSS, dh_meansDS, dh_stdDS, dh_meansSL1, dh_stdSL1, dh_meansDL1, dh_stdDL1])
        listI.append([i, di_means, di_std])
#########################################################
#8SPK case ; includes gender work
#########################################################
#mean values
        d80 = pd.DataFrame({"interp_point_dist": pd.Series(interpol_y)})
        d8=pd.concat([db_merged[['spk_x', 'spk_y','L1_x', 'L1_y', 'same_speaker', 'same_gender']],d80], axis=1)
        d8=d8[((d8['spk_x'] == c_it[0])|(d8['spk_x'] == c_it[1])|(d8['spk_x'] == c_it[2])|(d8['spk_x'] == c_it[3])|(d8['spk_x'] == c_it[4])|(d8['spk_x'] == c_it[5])|(d8['spk_x'] == c_it[6])|(d8['spk_x'] == c_it[7]))&\
        ((d8['spk_y'] == c_it[0])|(d8['spk_y'] == c_it[1])|(d8['spk_y'] == c_it[2])|(d8['spk_y'] == c_it[3])|(d8['spk_y'] == c_it[4])|(d8['spk_y'] == c_it[5])|(d8['spk_x'] == c_it[6])|(d8['spk_y'] == c_it[7]))]
        d8['same_L1'] =  d8['L1_x'] == d8['L1_y']
        d8_stack=np.vstack(d8['interp_point_dist'])
        d8_means = np.mean(d8_stack, axis=0)
        d8_std = np.std(d8_stack, axis=0)
#spk
        d8_stackSS=np.vstack(d8[d8['same_speaker']]['interp_point_dist'])
        d8_meansSS = np.mean(d8_stackSS, axis=0)
        d8_stdSS = np.std(d8_stackSS, axis=0)
        d8_stackDS=np.vstack(d8[~d8['same_speaker']]['interp_point_dist'])
        d8_meansDS = np.mean(d8_stackDS, axis=0)
        d8_stdDS = np.std(d8_stackDS, axis=0)
#L1
        d8_stackSL1=np.vstack(d8[d8['same_L1']]['interp_point_dist'])
        d8_meansSL1 = np.mean(d8_stackSL1, axis=0)
        d8_stdSL1 = np.std(d8_stackSL1, axis=0)
        d8_stackDL1=np.vstack(d8[~d8['same_L1']]['interp_point_dist'])
        d8_meansDL1 = np.mean(d8_stackDL1, axis=0)
        d8_stdDL1 = np.std(d8_stackDL1, axis=0)
#gender
        d8_stackSG=np.vstack(d8[d8['same_gender']]['interp_point_dist'])
        d8_meansSG = np.mean(d8_stackSG, axis=0)
        d8_stdSG = np.std(d8_stackSG, axis=0)
        d8_stackDG=np.vstack(d8[~d8['same_gender']]['interp_point_dist'])
        d8_meansDG = np.mean(d8_stackDG, axis=0)
        d8_stdDG = np.std(d8_stackDG, axis=0)

        listH8.append([i, d8_means, d8_std, d8_meansSS, d8_stdSS, d8_meansDS, d8_stdDS, d8_meansSL1, d8_stdSL1, d8_meansDL1, d8_stdDL1, d8_meansSG, d8_stdSG, d8_meansDG, d8_stdDG])

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
#SANITY CHECK
        print("sanity check...")
        print(dh.shape[0], db_merged.shape[0])
        de=pd.concat([db_merged,dh[['interp_point_dist']],di[['manhat_point_dist']]], axis=1)
        dg=pd.concat([dg,de], axis=0)
#    print("ici")
#    print(dg["point_dist"].iloc[1])
#    print(dg["point_dist"].shape[0])
#    print(dg["point_dist"].shape[1])
#    print("ici")
#    dg["interpolated_point_dist"] = interpolate_to_longest_mesh(dg["point_dist"].tolist(), common_x)
#    dg['interpolated_point_dist'] = dg["point_dist"].apply(interpolate_to_new_mesh)
#    print(dg['interpolated_point_dist'].shape)
#add new observables
    dg['Grad_Norm']=np.sqrt(dg['MrowG']**2 + dg['McolG']**2)

#file info
    print("categories : ",list(dg))
    print("nombre de lignes : ",dg.shape[0])
    os.chdir(args.out_dir)
    DI=pd.DataFrame(listI, columns=["LAY", "mean_point_dist", "std_point_dist"])
    DH=pd.DataFrame(listH, columns=["LAY", "mean_point_dist", "std_point_dist", "mean_point_SS", "std_point_SS", "mean_point_DS", "std_point_DS", "mean_point_SL1", "std_point_SL1", "mean_point_DL1", "std_point_DL1"])
    D8=pd.DataFrame(listH8, columns=["LAY", "mean_point_dist", "std_point_dist", "mean_point_SS", "std_point_SS", "mean_point_DS", "std_point_DS", "mean_point_SL1", "std_point_SL1", "mean_point_DL1", "std_point_DL1", "mean_point_SG", "std_point_SG", "mean_point_DG", "std_point_DG"])
    DA=pd.DataFrame(listA, columns=["LAY", f"mean_point_dist_{c_it[0]}", f"std_point_dist_{c_it[0]}", f"mean_point_dist_{c_it[1]}", f"std_point_dist_{c_it[1]}", f"mean_point_dist_{c_it[2]}", f"std_point_dist_{c_it[2]}", f"mean_point_dist_{c_it[3]}", f"std_point_dist_{c_it[3]}", f"mean_point_dist_{c_it[4]}", f"std_point_dist_{c_it[4]}", f"mean_point_dist_{c_it[5]}", f"std_point_dist_{c_it[5]}", f"mean_point_dist_{c_it[6]}", f"std_point_dist_{c_it[6]}", f"mean_point_dist_{c_it[7]}", f"std_point_dist_{c_it[7]}"])
    print(list(DA))
#dataframe saved
    subDG=dg[["layer","cost_LD","Ncost_LD","spk_x", "file_x",  "L1_x", "age_x", "sex_x", "level_fr_x", "level_ru_x", "spk_y", "file_y", "L1_y", "age_y", "sex_y", "level_fr_y", "level_ru_y",\
            "R", "len_devxy", "m", "n", "dist_A", "dist_N","Grad_Norm", "interpolated_audio_path","audio_cost_LD","Naudio_cost_LD",\
            "frechet_dist","point_dist","manh_dist"]]
    print("writing to output...")
    argout = str(work_file).replace("table","result").replace("layer_24-","")
    argout2 = str(work_file).replace("table","meanval_19spk").replace("layer_24-","")
    argout3 = str(work_file).replace("table","meanval_8spk").replace("layer_24-","")
    argout4 = str(work_file).replace("table","meanval_indspk").replace("layer_24-","")
    argout5 = str(work_file).replace("table","mean-manhdist_19SPK").replace("layer_24-","")
    subDG.to_pickle(argout)
    DH.to_pickle(argout2)
    D8.to_pickle(argout3)
    DA.to_pickle(argout4)
    DI.to_pickle(argout5)
#non averaged graphical views
    #print("s1")
#    print("s2")
#    print("s3")
#    print(len(new_x_mesh))
#    print(new_x_mesh)
    lines = ["-","--","-.",":","."]
    linecycler = cycle(lines)
#19SPK
    fig = plt.figure(figsize=(6, 8))
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    for ix, rx in DH.iterrows():
        lbl=rx["LAY"]
        x_value=new_x_mesh
        y_value=rx["mean_point_dist"]
        z_value=rx["std_point_dist"]
        plt.plot(x_value,y_value, next(linecycler),linewidth=0.5,label=f"layer {lbl}")
        plt.fill_between(x_value, y_value - z_value, y_value + z_value, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1.1), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_mean-eucl-dist_19SPK.pdf", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6, 8))
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Manhattan dist.')
    for ix, rx in DI.iterrows():
        lbl=rx["LAY"]
        x_value=new_x_mesh
        y_value=rx["mean_point_dist"]
        z_value=rx["std_point_dist"]
        plt.plot(x_value,y_value, next(linecycler),linewidth=0.5,label=f"layer {lbl}")
        plt.fill_between(x_value, y_value - z_value, y_value + z_value, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.02, 1.1), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_mean-manh-dist_19SPK.pdf", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6, 8))
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    x_value=new_x_mesh
    y_value1=DH["mean_point_SS"].iloc[16]
    z_value1=DH["std_point_SS"].iloc[16]
    y_value2=DH["mean_point_DS"].iloc[16]
    z_value2=DH["std_point_DS"].iloc[16]
    plt.plot(x_value,y_value1, "-",linewidth=0.5,label="same speaker")
    plt.fill_between(x_value, y_value1 - z_value1, y_value1 + z_value1, alpha=0.1)
    plt.plot(x_value,y_value2, "--",linewidth=0.5,label="diff_speaker")
    plt.fill_between(x_value, y_value2 - z_value2, y_value2 + z_value2, alpha=0.1)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_spk-eff_19SPK_L16.pdf", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6, 8))
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    x_value=new_x_mesh
    y_value1=DH["mean_point_SL1"].iloc[16]
    z_value1=DH["std_point_SL1"].iloc[16]
    y_value2=DH["mean_point_DL1"].iloc[16]
    z_value2=DH["std_point_DL1"].iloc[16]
    plt.plot(x_value,y_value1, "-",linewidth=0.5,label="same_L1")
    plt.fill_between(x_value, y_value1 - z_value1, y_value1 + z_value1, alpha=0.1)
    plt.plot(x_value,y_value2, "--",linewidth=0.5,label="diff_L1")
    plt.fill_between(x_value, y_value2 - z_value2, y_value2 + z_value2, alpha=0.1)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_L1-eff_19SPK_L16.pdf", bbox_inches='tight')
    plt.close()
#8SPK
    fig = plt.figure(figsize=(6, 8))
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    for ix, rx in D8.iterrows():
        lbl=rx["LAY"]
        x_value=new_x_mesh
        y_value=rx["mean_point_dist"]
        z_value=rx["std_point_dist"]
        plt.plot(x_value,y_value, next(linecycler),linewidth=0.5,label=f"layer {lbl}")
        plt.fill_between(x_value, y_value - z_value, y_value + z_value, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.02, 0.6), loc="upper left", ncol=2)
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_mean-eucl-dist_8SPK.pdf", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6, 8))
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    x_value=new_x_mesh
    y_value1=D8["mean_point_SS"].iloc[16]
    z_value1=D8["std_point_SS"].iloc[16]
    y_value2=D8["mean_point_DS"].iloc[16]
    z_value2=D8["std_point_DS"].iloc[16]
    plt.plot(x_value,y_value1, next(linecycler),linewidth=0.5,label="same speaker")
    plt.fill_between(x_value, y_value1 - z_value1, y_value1 + z_value1, alpha=0.1)
    plt.plot(x_value,y_value2, next(linecycler),linewidth=0.5,label="diff_speaker")
    plt.fill_between(x_value, y_value2 - z_value2, y_value2 + z_value2, alpha=0.1)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_spk-eff_8SPK_L16.pdf", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6, 8))
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    x_value=new_x_mesh
    y_value1=D8["mean_point_SL1"].iloc[16]
    z_value1=D8["std_point_SL1"].iloc[16]
    y_value2=D8["mean_point_DL1"].iloc[16]
    z_value2=D8["std_point_DL1"].iloc[16]
    plt.plot(x_value,y_value1, next(linecycler),linewidth=0.5,label="same_L1")
    plt.fill_between(x_value, y_value1 - z_value1, y_value1 + z_value1, alpha=0.1)
    plt.plot(x_value,y_value2, next(linecycler),linewidth=0.5,label="diff_L1")
    plt.fill_between(x_value, y_value2 - z_value2, y_value2 + z_value2, alpha=0.1)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_L1-eff_8SPK_L16.pdf", bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(6, 8))
    plt.xlabel('t_norm')
    plt.ylabel('point-by-point Eucl. dist.')
    x_value=new_x_mesh
    y_value1=D8["mean_point_SG"].iloc[16]
    z_value1=D8["std_point_SG"].iloc[16]
    y_value2=D8["mean_point_DG"].iloc[16]
    z_value2=D8["std_point_DG"].iloc[16]
    plt.plot(x_value,y_value1, next(linecycler),linewidth=0.5,label="same_gender")
    plt.fill_between(x_value, y_value1 - z_value1, y_value1 + z_value1, alpha=0.1)
    plt.plot(x_value,y_value2, next(linecycler),linewidth=0.5,label="diff_gender")
    plt.fill_between(x_value, y_value2 - z_value2, y_value2 + z_value2, alpha=0.1)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_gend-eff_8SPK_L16.pdf", bbox_inches='tight')
    plt.close()

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
    plt.legend(bbox_to_anchor=(1.02, 0.8), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_eff_indiv_L16.pdf", bbox_inches='tight')
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
    plt.legend(bbox_to_anchor=(1.02, 0.8), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_eff_indiv_L19.pdf", bbox_inches='tight')
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
    plt.legend(bbox_to_anchor=(1.02, 0.8), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_eff_indiv_L20.pdf", bbox_inches='tight')
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
    plt.legend(bbox_to_anchor=(1.02, 0.8), loc="upper left")
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_eucl-dist_eff_indiv_L5.pdf", bbox_inches='tight')
    plt.legend()
    plt.close()
#SSanovas
    #dg_mean = dg.groupby("layer")["dist_A"].mean()
    #dg_std = dg.groupby("layer")["dist_A"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["dist_A"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["dist_A"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["dist_A"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["dist_A"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_A')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "+-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "x-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_dist_A_19SPK.pdf", bbox_inches='tight')
    plt.close()

    #dg_mean = dg.groupby("layer")["dist_N"].mean()
    #dg_std = dg.groupby("layer")["dist_N"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["dist_N"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["dist_N"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["dist_N"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["dist_N"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_N')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.title("Normalized distance (vector DTW path)")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_dist_N_19SPK.pdf", bbox_inches='tight')
    plt.close()

    #dg_mean = dg.groupby("layer")["Grad_Norm"].mean()
    #dg_std = dg.groupby("layer")["Grad_Norm"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["Grad_Norm"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["Grad_Norm"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["Grad_Norm"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["Grad_Norm"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Max_Grad_Norm')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.title("Maximum Gradient Norm")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_Max_Grad_Norm_19SPK.pdf", bbox_inches='tight')
    plt.close()
    #dg_mean = dg.groupby("layer")["R"].mean()
    #dg_std = dg.groupby("layer")["R"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["R"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["R"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["R"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["R"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation ratio')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.title("Deviation-to-Diagonal events ratio")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_dev_ratio_19SPK.pdf", bbox_inches='tight')
    plt.close()
#ABSOLUTE COST
#COST MATRICES
# vect cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanDS =  dg[dg['same_speaker'] == False].groupby("layer")["cost_LD"].mean()
    M1= dg_meanDS.iloc[20]
    dg_stdDS =  dg[dg['same_speaker'] == False].groupby("layer")["cost_LD"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["cost_LD"].mean()
    M2=dg_meanSS.iloc[20]
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["cost_LD"].std()
# audio cost matrix
    dg_meanDSref = dg[dg['same_speaker'] == False].groupby("layer")["audio_cost_LD"].mean()
    dg_stdDSref = dg[dg['same_speaker'] == False].groupby("layer")["audio_cost_LD"].std()
    m1=dg_meanDSref.iloc[20]
    dg_meanSSref = dg[dg['same_speaker'] == True].groupby("layer")["audio_cost_LD"].mean()
    dg_stdSSref = dg[dg['same_speaker'] == True].groupby("layer")["audio_cost_LD"].std()
    m2=dg_meanSSref.iloc[20]
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('vect_cost_mat(-1;-1)')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.plot(dg_meanSSref.index, dg_meanSSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (same_spk))")
    plt.plot(dg_meanDSref.index, dg_meanDSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (diff_spk)")
    plt.legend()
    plt.title('Last digit of the DTW vector cost matrix, compared w. audio DTW')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_DTW_COST_LAST_DIGIT_19SPK.pdf", bbox_inches='tight')
    plt.close()
#NORMALIZED COST
#COST MATRICES
# audio cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanDS =  dg[dg['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
    M1= dg_meanDS.iloc[20]
    dg_stdDS =  dg[dg['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
    M2=dg_meanSS.iloc[20]
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# vect cost matrix
    dg_meanDSref = dg[dg['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
    dg_stdDSref = dg[dg['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].std()
    m1=dg_meanDSref.iloc[20]
    dg_meanSSref = dg[dg['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
    dg_stdSSref = dg[dg['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].std()
    m2=dg_meanSSref.iloc[20]
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('norm_vect_cost_mat(-1;-1)')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.plot(dg_meanSSref.index, dg_meanSSref.values*((M1+M2)/(m1+m2)), "--",label="norm-Audio cost (same_spk))")
    plt.plot(dg_meanDSref.index, dg_meanDSref.values*((M1+M2)/(m1+m2)), "--",label="norm-Audio cost (diff_spk)")
    plt.legend()
    plt.title('Last digit of the DTW vector cost matrix (normalized), compared w. audio DTW')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_DTW_NORM_COST_LAST_DIGIT_19SPK.pdf", bbox_inches='tight')
    plt.close()
#count the number of deviations
    #dg_mean = dg.groupby("layer")["len_devxy"].mean()
    #dg_std = dg.groupby("layer")["len_devxy"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["len_devxy"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["len_devxy"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["len_devxy"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["len_devxy"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation count')
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    plt.title('Layer-wise Count of Deviations-to-Diagonal events')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_DEV_COUNT_19SPK.pdf", bbox_inches='tight')
    plt.close()

#Frechet distance
    #dg_mean = dg.groupby("layer")["frechet_dist"].mean()
    #dg_std = dg.groupby("layer")["frechet_dist"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["frechet_dist"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["frechet_dist"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["frechet_dist"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["frechet_dist"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Frechet distance')
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.title('Vector Path to Audio Path Frechet distance')
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_FRECHET_19SPK.pdf", bbox_inches='tight')
    plt.close()
###############################################################################################################""
    #distinguer selon la L1 du locuteur également
    #dg_mean = dg.groupby("layer")["dist_A"].mean()
    #dg_std = dg.groupby("layer")["dist_A"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["dist_A"].mean()
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["dist_A"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["dist_A"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["dist_A"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["dist_A"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["dist_A"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_A')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.title("Absolute distance (vector DTW path)")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_dist_A_19SPK.pdf", bbox_inches='tight')
    plt.close()
    #dg_mean = dg.groupby("layer")["dist_N"].mean()
    #dg_std = dg.groupby("layer")["dist_N"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["dist_N"].mean()
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["dist_N"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["dist_N"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["dist_N"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["dist_N"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["dist_N"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_N')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.title("Normalized distance (vector DTW path)")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_dist_N_19SPK.pdf", bbox_inches='tight')
    plt.close()
# Gradients
    #dg_mean = dg.groupby("layer")["Grad_Norm"].mean()
    #dg_std = dg.groupby("layer")["Grad_Norm"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["Grad_Norm"].mean()
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["Grad_Norm"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["Grad_Norm"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["Grad_Norm"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Grad_Norm"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Grad_Norm"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Max_Grad_Norm')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_Max_Grad_Norm_19SPK.pdf", bbox_inches='tight')
    plt.close()
#   deviation ratio
    #dg_mean = dg.groupby("layer")["R"].mean()
    #dg_std = dg.groupby("layer")["R"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["R"].mean()
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["R"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["R"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["R"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["R"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["R"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation ratio')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_dev_ratio_19SPK.pdf", bbox_inches='tight')
    plt.close()
#ABSOLUTE COST
#COST MATRICES
# audio cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["cost_LD"].mean()
    M1= dg_meanR.iloc[20]
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["cost_LD"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["cost_LD"].mean()
    M2=dg_meanF.iloc[20]
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["cost_LD"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["cost_LD"].mean()
    M3=dg_meanV.iloc[20]
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["cost_LD"].std()
# vect cost matrix
    dg_meanRref = dg[dg['L1'] == 'ru'].groupby("layer")["audio_cost_LD"].mean()
    dg_stdRref = dg[dg['L1'] == 'ru'].groupby("layer")["audio_cost_LD"].std()
    m1=dg_meanRref.iloc[20]
    dg_meanFref = dg[dg['L1'] == 'fr'].groupby("layer")["audio_cost_LD"].mean()
    dg_stdFref = dg[dg['L1'] == 'fr'].groupby("layer")["audio_cost_LD"].std()
    m2=dg_meanFref.iloc[20]
    dg_meanVref = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["audio_cost_LD"].mean()
    dg_stdVref = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["audio_cost_LD"].std()
    m3=dg_meanVref.iloc[20]
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('vect_cost_mat(-1;-1)')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="Vector cost (ru only)")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="Vector cost (fr only)")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="Vector cost (fr vs ru)")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.plot(dg_meanRref.index, dg_meanRref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (ru only)")
    plt.plot(dg_meanFref.index, dg_meanFref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (fr only)")
    plt.plot(dg_meanVref.index, dg_meanVref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (fr vs ru)")
    plt.legend()
    plt.title('Last digit of the DTW vector cost matrix, compared w. audio DTW')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_DTW_COST_LAST_DIGIT_19SPK.pdf", bbox_inches='tight')
    plt.close()
#NORMALIZED COST
#COST MATRICES
# audio cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
    M1= dg_meanR.iloc[20]
    ST1= dg_stdR.iloc[20]
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
    M2=dg_meanF.iloc[20]
    ST2=dg_stdF.iloc[20]
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
    M3=dg_meanV.iloc[20]
    ST3=dg_stdV.iloc[20]
    print("L1 effect Ncost means and SD :")
    print(M1, ST1, M2, ST2, M3, ST3)

# vect cost matrix
    dg_meanRref = dg[dg['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
    dg_stdRref = dg[dg['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].std()
    m1=dg_meanRref.iloc[20]
    st1=dg_stdRref.iloc[20]
    dg_meanFref = dg[dg['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
    dg_stdFref = dg[dg['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].std()
    m2=dg_meanFref.iloc[20]
    st2=dg_stdFref.iloc[20]
    dg_meanVref = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
    dg_stdVref = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].std()
    m3=dg_meanVref.iloc[20]
    st3=dg_stdVref.iloc[20]
    print("L1 effect Naudiocost means and SD :")
    print(m1, st1, m2, st2, m3, st3)
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="Vector cost (ru only)")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="Vector cost (fr only)")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="Vector cost (fr vs ru)")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.plot(dg_meanRref.index, dg_meanRref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (ru only)")
    plt.plot(dg_meanFref.index, dg_meanFref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (fr only)")
    plt.plot(dg_meanVref.index, dg_meanVref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (fr vs ru)")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_DTW_NORM_COST_LAST_DIGIT_19SPK.pdf", bbox_inches='tight')
    plt.close()
#count the number of deviations
    #dg_mean = dg.groupby("layer")["len_devxy"].mean()
    #dg_std = dg.groupby("layer")["len_devxy"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["len_devxy"].mean()
    print(dg_meanR)
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["len_devxy"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["len_devxy"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["len_devxy"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["len_devxy"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["len_devxy"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation count')
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.legend()
    plt.title('Layer-wise Count of Deviations-to-Diagonal events')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_DEV_COUNT_19SPK.pdf", bbox_inches='tight')
    plt.close()

#Frechet distance
    #dg_mean = dg.groupby("layer")["frechet_dist"].mean()
    #dg_std = dg.groupby("layer")["frechet_dist"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["frechet_dist"].mean()
    print(dg_meanR)
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["frechet_dist"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["frechet_dist"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["frechet_dist"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["frechet_dist"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["frechet_dist"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Frechet distance')
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.legend()
    plt.title('Vector Path to Audio Path Frechet distance')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_FRECHET_19SPK.pdf", bbox_inches='tight')
    plt.close()
###############################################################################################################
    #8SPK : cas avec un sous corpus équilibré en genre (m/f) et en L1 (ru/fr)
    dg=dg[((dg['spk_x'] == c_it[0])|(dg['spk_x'] == c_it[1])|(dg['spk_x'] == c_it[2])|(dg['spk_x'] == c_it[3])|(dg['spk_x'] == c_it[4])|(dg['spk_x'] == c_it[5])|(dg['spk_x'] == c_it[6])|(dg['spk_x'] == c_it[7]))&\
        ((dg['spk_y'] == c_it[0])|(dg['spk_y'] == c_it[1])|(dg['spk_y'] == c_it[2])|(dg['spk_y'] == c_it[3])|(dg['spk_y'] == c_it[4])|(dg['spk_y'] == c_it[5])|(dg['spk_x'] == c_it[6])|(dg['spk_y'] == c_it[7]))]

    #dg_mean = dg.groupby("layer")["dist_A"].mean()
    #dg_std = dg.groupby("layer")["dist_A"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["dist_A"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["dist_A"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["dist_A"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["dist_A"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_A')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "+-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "x-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_dist_A_8SPK.pdf", bbox_inches='tight')
    plt.close()

    #dg_mean = dg.groupby("layer")["dist_N"].mean()
    #dg_std = dg.groupby("layer")["dist_N"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["dist_N"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["dist_N"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["dist_N"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["dist_N"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_N')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.title("Normalized distance (vector DTW path)")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_dist_N_8SPK.pdf", bbox_inches='tight')
    plt.close()

    #dg_mean = dg.groupby("layer")["Grad_Norm"].mean()
    #dg_std = dg.groupby("layer")["Grad_Norm"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["Grad_Norm"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["Grad_Norm"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["Grad_Norm"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["Grad_Norm"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Max_Grad_Norm')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_Max_Grad_Norm_8SPK.pdf", bbox_inches='tight')
    plt.close()

    #dg_mean = dg.groupby("layer")["R"].mean()
    #dg_std = dg.groupby("layer")["R"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["R"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["R"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["R"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["R"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation ratio')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.title("Deviation-to-Diagonal events ratio")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_dev_ratio_8SPK.pdf", bbox_inches='tight')
    plt.close()
#ABSOLUTE COST
#COST MATRICES
# audio cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanDS =  dg[dg['same_speaker'] == False].groupby("layer")["cost_LD"].mean()
    M1= dg_meanDS.iloc[20]
    dg_stdDS =  dg[dg['same_speaker'] == False].groupby("layer")["cost_LD"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["cost_LD"].mean()
    M2=dg_meanSS.iloc[20]
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["cost_LD"].std()
# vect cost matrix
    dg_meanDSref = dg[dg['same_speaker'] == False].groupby("layer")["audio_cost_LD"].mean()
    m1=dg_meanDSref.iloc[20]
    dg_meanSSref = dg[dg['same_speaker'] == True].groupby("layer")["audio_cost_LD"].mean()
    m2=dg_meanSSref.iloc[20]
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('vect_cost_mat(-1;-1)')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.plot(dg_meanSSref.index, dg_meanSSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (same_spk))")
    plt.plot(dg_meanDSref.index, dg_meanDSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (diff_spk)")
    plt.legend()
    plt.title('Last digit of the DTW vector cost matrix, compared w. audio DTW')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_DTW_COST_LAST_DIGIT_8SPK.pdf", bbox_inches='tight')
    plt.close()
#NORMALIZED COST
#COST MATRICES
# audio cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanDS =  dg[dg['same_speaker'] == False].groupby("layer")["Ncost_LD"].mean()
    M1= dg_meanDS.iloc[20]
    dg_stdDS =  dg[dg['same_speaker'] == False].groupby("layer")["Ncost_LD"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["Ncost_LD"].mean()
    M2=dg_meanSS.iloc[20]
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["Ncost_LD"].std()
# vect cost matrix
    dg_meanDSref = dg[dg['same_speaker'] == False].groupby("layer")["Naudio_cost_LD"].mean()
    m1=dg_meanDSref.iloc[20]
    dg_meanSSref = dg[dg['same_speaker'] == True].groupby("layer")["Naudio_cost_LD"].mean()
    m2=dg_meanSSref.iloc[20]
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('norm_vect_cost_mat(-1;-1)')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.plot(dg_meanSSref.index, dg_meanSSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (same_spk))")
    plt.plot(dg_meanDSref.index, dg_meanDSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (diff_spk)")
    plt.legend()
    plt.title('vector DTW vs. audio DTW cost (last element, normalized)')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_DTW_NORM_COST_LAST_DIGIT_8SPK.pdf", bbox_inches='tight')
    plt.close()
#count the number of deviations
    #dg_mean = dg.groupby("layer")["len_devxy"].mean()
    #dg_std = dg.groupby("layer")["len_devxy"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["len_devxy"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["len_devxy"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["len_devxy"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["len_devxy"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation count')
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    plt.title('Layer-wise Count of Deviations-to-Diagonal events')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_DEV_COUNT_8SPK.pdf", bbox_inches='tight')
    plt.close()

#Frechet distance
    #dg_mean = dg.groupby("layer")["frechet_dist"].mean()
    #dg_std = dg.groupby("layer")["frechet_dist"].std()
    dg_meanDS = dg[dg['same_speaker'] == False].groupby("layer")["frechet_dist"].mean()
    dg_stdDS = dg[dg['same_speaker'] == False].groupby("layer")["frechet_dist"].std()
    dg_meanSS = dg[dg['same_speaker'] == True].groupby("layer")["frechet_dist"].mean()
    dg_stdSS = dg[dg['same_speaker'] == True].groupby("layer")["frechet_dist"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Frechet distance')
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different speakers")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same speaker")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    plt.title('Vector Path to Audio Path Frechet distance')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_spk_effect_FRECHET_8SPK.pdf", bbox_inches='tight')
    plt.close()
###############################################################################################################
    #distinguer selon le genre du locuteur
    #dg_mean = dg.groupby("layer")["dist_A"].mean()
    #dg_std = dg.groupby("layer")["dist_A"].std()
    dg_meanDS = dg[dg['same_gender'] == False].groupby("layer")["dist_A"].mean()
    dg_stdDS = dg[dg['same_gender'] == False].groupby("layer")["dist_A"].std()
    dg_meanSS = dg[dg['same_gender'] == True].groupby("layer")["dist_A"].mean()
    dg_stdSS = dg[dg['same_gender'] == True].groupby("layer")["dist_A"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_A')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "+-",label="different gender")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "x-",label="same gender")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_gend_effect_dist_A_8SPK.pdf", bbox_inches='tight')
    plt.close()

    #dg_mean = dg.groupby("layer")["dist_N"].mean()
    #dg_std = dg.groupby("layer")["dist_N"].std()
    dg_meanDS = dg[dg['same_gender'] == False].groupby("layer")["dist_N"].mean()
    dg_stdDS = dg[dg['same_gender'] == False].groupby("layer")["dist_N"].std()
    dg_meanSS = dg[dg['same_gender'] == True].groupby("layer")["dist_N"].mean()
    dg_stdSS = dg[dg['same_gender'] == True].groupby("layer")["dist_N"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_N')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different gender")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same gender")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.title("Normalized distance (vector DTW path)")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_gend_effect_dist_N_8SPK.pdf", bbox_inches='tight')
    plt.close()

    #dg_mean = dg.groupby("layer")["Grad_Norm"].mean()
    #dg_std = dg.groupby("layer")["Grad_Norm"].std()
    dg_meanDS = dg[dg['same_gender'] == False].groupby("layer")["Grad_Norm"].mean()
    dg_stdDS = dg[dg['same_gender'] == False].groupby("layer")["Grad_Norm"].std()
    dg_meanSS = dg[dg['same_gender'] == True].groupby("layer")["Grad_Norm"].mean()
    dg_stdSS = dg[dg['same_gender'] == True].groupby("layer")["Grad_Norm"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Max_Grad_Norm')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different gender")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same gender")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_gend_effect_Max_Grad_Norm_8SPK.pdf", bbox_inches='tight')
    plt.close()

    #dg_mean = dg.groupby("layer")["R"].mean()
    #dg_std = dg.groupby("layer")["R"].std()
    dg_meanDS = dg[dg['same_gender'] == False].groupby("layer")["R"].mean()
    dg_stdDS = dg[dg['same_gender'] == False].groupby("layer")["R"].std()
    dg_meanSS = dg[dg['same_gender'] == True].groupby("layer")["R"].mean()
    dg_stdSS = dg[dg['same_gender'] == True].groupby("layer")["R"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation ratio')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different gender")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same gender")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_gend_effect_dev_ratio_8SPK.pdf", bbox_inches='tight')
    plt.close()
#ABSOLUTE COST
#COST MATRICES
# audio cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanDS =  dg[dg['same_gender'] == False].groupby("layer")["cost_LD"].mean()
    M1= dg_meanDS.iloc[20]
    dg_stdDS =  dg[dg['same_gender'] == False].groupby("layer")["cost_LD"].std()
    dg_meanSS = dg[dg['same_gender'] == True].groupby("layer")["cost_LD"].mean()
    M2=dg_meanSS.iloc[20]
    dg_stdSS = dg[dg['same_gender'] == True].groupby("layer")["cost_LD"].std()
# vect cost matrix
    dg_meanDSref = dg[dg['same_gender'] == False].groupby("layer")["audio_cost_LD"].mean()
    m1=dg_meanDSref.iloc[20]
    dg_meanSSref = dg[dg['same_gender'] == True].groupby("layer")["audio_cost_LD"].mean()
    m2=dg_meanSSref.iloc[20]
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('vect_cost_mat(-1;-1)')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different gender")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same gender")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.plot(dg_meanSSref.index, dg_meanSSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (same_spk))")
    plt.plot(dg_meanDSref.index, dg_meanDSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (diff_spk)")
    plt.legend()
    plt.title('Last digit of the DTW vector cost matrix, compared w. audio DTW')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_gend_effect_DTW_COST_LAST_DIGIT_8SPK.pdf", bbox_inches='tight')
    plt.close()
#NORMALIZED COST
#COST MATRICES
# audio cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanDS =  dg[dg['same_gender'] == False].groupby("layer")["Ncost_LD"].mean()
    M1= dg_meanDS.iloc[20]
    dg_stdDS =  dg[dg['same_gender'] == False].groupby("layer")["Ncost_LD"].std()
    dg_meanSS = dg[dg['same_gender'] == True].groupby("layer")["Ncost_LD"].mean()
    M2=dg_meanSS.iloc[20]
    dg_stdSS = dg[dg['same_gender'] == True].groupby("layer")["Ncost_LD"].std()
# vect cost matrix
    dg_meanDSref = dg[dg['same_gender'] == False].groupby("layer")["Naudio_cost_LD"].mean()
    m1=dg_meanDSref.iloc[20]
    dg_meanSSref = dg[dg['same_gender'] == True].groupby("layer")["Naudio_cost_LD"].mean()
    m2=dg_meanSSref.iloc[20]
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('norm_vect_cost_mat(-1;-1)')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different gender")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same gender")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.plot(dg_meanSSref.index, dg_meanSSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (same_spk))")
    plt.plot(dg_meanDSref.index, dg_meanDSref.values*((M1+M2)/(m1+m2)), "--",label="Audio cost (diff_spk)")
    plt.legend()
    plt.title('Last digit of the DTW vector cost matrix (normalized), compared w. audio DTW')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_gend_effect_DTW_NORM_COST_LAST_DIGIT_8SPK.pdf", bbox_inches='tight')
    plt.close()
#count the number of deviations
    #dg_mean = dg.groupby("layer")["len_devxy"].mean()
    #dg_std = dg.groupby("layer")["len_devxy"].std()
    dg_meanDS = dg[dg['same_gender'] == False].groupby("layer")["len_devxy"].mean()
    dg_stdDS = dg[dg['same_gender'] == False].groupby("layer")["len_devxy"].std()
    dg_meanSS = dg[dg['same_gender'] == True].groupby("layer")["len_devxy"].mean()
    dg_stdSS = dg[dg['same_gender'] == True].groupby("layer")["len_devxy"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation count')
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different gender")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same_gender")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    plt.title('Layer-wise Count of Deviations-to-Diagonal events')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_gend_effect_DEV_COUNT_8SPK.pdf", bbox_inches='tight')
    plt.close()

#Frechet distance
    #dg_mean = dg.groupby("layer")["frechet_dist"].mean()
    #dg_std = dg.groupby("layer")["frechet_dist"].std()
    dg_meanDS = dg[dg['same_gender'] == False].groupby("layer")["frechet_dist"].mean()
    dg_stdDS = dg[dg['same_gender'] == False].groupby("layer")["frechet_dist"].std()
    dg_meanSS = dg[dg['same_gender'] == True].groupby("layer")["frechet_dist"].mean()
    dg_stdSS = dg[dg['same_gender'] == True].groupby("layer")["frechet_dist"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Frechet distance')
    plt.plot(dg_meanDS.index, dg_meanDS.values, "x-",label="different gender")
    plt.fill_between(dg_meanDS.index, dg_meanDS.values - dg_stdDS.values, dg_meanDS.values + dg_stdDS.values, alpha=0.3)
    plt.plot(dg_meanSS.index, dg_meanSS.values, "+-",label="same gender")
    plt.fill_between(dg_meanSS.index, dg_meanSS.values - dg_stdSS.values, dg_meanSS.values + dg_stdSS.values, alpha=0.3)
    plt.legend()
    plt.title('Vector Path to Audio Path Frechet distance')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_gend_effect_FRECHET_8SPK.pdf", bbox_inches='tight')
    plt.close()
###############################################################################################################
    #distinguer selon la L1 du locuteur également
    dg["L1"] = dg.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
    dg_mean = dg.groupby("layer")["dist_A"].mean()
    dg_std = dg.groupby("layer")["dist_A"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["dist_A"].mean()
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["dist_A"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["dist_A"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["dist_A"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["dist_A"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["dist_A"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_A')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_dist_A_8SPK.pdf", bbox_inches='tight')
    plt.close()
    #dg_mean = dg.groupby("layer")["dist_N"].mean()
    #dg_std = dg.groupby("layer")["dist_N"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["dist_N"].mean()
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["dist_N"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["dist_N"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["dist_N"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["dist_N"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["dist_N"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('dist_N')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.title("Normalized distance (vector DTW path)")
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_dist_N_8SPK.pdf", bbox_inches='tight')
    plt.close()
# Gradients
    #dg_mean = dg.groupby("layer")["Grad_Norm"].mean()
    #dg_std = dg.groupby("layer")["Grad_Norm"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["Grad_Norm"].mean()
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["Grad_Norm"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["Grad_Norm"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["Grad_Norm"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Grad_Norm"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Grad_Norm"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Max_Grad_Norm')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_Max_Grad_Norm_8SPK.pdf", bbox_inches='tight')
    plt.close()
#   deviation ratio
    #dg_mean = dg.groupby("layer")["R"].mean()
    #dg_std = dg.groupby("layer")["R"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["R"].mean()
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["R"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["R"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["R"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["R"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["R"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation ratio')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.legend()
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_dev_ratio_8SPK.pdf", bbox_inches='tight')
    plt.close()
#ABSOLUTE COST
#COST MATRICES
# audio cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["cost_LD"].mean()
    M1= dg_meanR.iloc[20]
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["cost_LD"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["cost_LD"].mean()
    M2=dg_meanF.iloc[20]
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["cost_LD"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["cost_LD"].mean()
    M3=dg_meanV.iloc[20]
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["cost_LD"].std()
# vect cost matrix
    dg_meanRref = dg[dg['L1'] == 'ru'].groupby("layer")["audio_cost_LD"].mean()
    m1=dg_meanRref.iloc[20]
    dg_meanFref = dg[dg['L1'] == 'fr'].groupby("layer")["audio_cost_LD"].mean()
    m2=dg_meanFref.iloc[20]
    dg_meanVref = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["audio_cost_LD"].mean()
    m3=dg_meanVref.iloc[20]
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('vect_cost_mat(-1;-1)')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.plot(dg_meanRref.index, dg_meanRref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (ru only)")
    plt.plot(dg_meanFref.index, dg_meanFref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (fr only)")
    plt.plot(dg_meanVref.index, dg_meanVref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (fr vs ru)")
    plt.legend()
    plt.title('Last digit of the DTW vector cost matrix, compared w. audio DTW')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_DTW_COST_LAST_DIGIT_8SPK.pdf", bbox_inches='tight')
    plt.close()
#NORMALIZED COST
#COST MATRICES
# audio cost matrix
    #dg_mean = dg.groupby("layer")["cost_LD"].mean()
    #dg_std = dg.groupby("layer")["cost_LD"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["Ncost_LD"].mean()
    M1= dg_meanR.iloc[20]
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["Ncost_LD"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["Ncost_LD"].mean()
    M2=dg_meanF.iloc[20]
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["Ncost_LD"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].mean()
    M3=dg_meanV.iloc[20]
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Ncost_LD"].std()
# vect cost matrix
    dg_meanRref = dg[dg['L1'] == 'ru'].groupby("layer")["Naudio_cost_LD"].mean()
    m1=dg_meanRref.iloc[20]
    dg_meanFref = dg[dg['L1'] == 'fr'].groupby("layer")["Naudio_cost_LD"].mean()
    m2=dg_meanFref.iloc[20]
    dg_meanVref = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["Naudio_cost_LD"].mean()
    m3=dg_meanVref.iloc[20]
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('norm_vect_cost_mat(-1;-1)')
#    plt.plot(dg_mean.index, dg_mean.values, "o-",label="all speakers")
#    plt.fill_between(dg_mean.index, dg_mean.values - dg_std.values, dg_mean.values + dg_std.values, alpha=0.3)
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.plot(dg_meanRref.index, dg_meanRref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (ru only)")
    plt.plot(dg_meanFref.index, dg_meanFref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (fr only)")
    plt.plot(dg_meanVref.index, dg_meanVref.values*((M1+M2+M3)/(m1+m2+m3)), "--",label="Audio cost (fr vs ru)")
    plt.legend()
    plt.title('Last digit of the DTW vector cost matrix (normalized), compared w. audio DTW')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_DTW_NORM_COST_LAST_DIGIT_8SPK.pdf", bbox_inches='tight')
    plt.close()
#count the number of deviations
    #dg_mean = dg.groupby("layer")["len_devxy"].mean()
    #dg_std = dg.groupby("layer")["len_devxy"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["len_devxy"].mean()
    print(dg_meanR)
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["len_devxy"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["len_devxy"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["len_devxy"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["len_devxy"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["len_devxy"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('deviation count')
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.legend()
    plt.title('Layer-wise Count of Deviations-to-Diagonal events')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_DEV_COUNT_8SPK.pdf", bbox_inches='tight')
    plt.close()

#Frechet distance
    #dg_mean = dg.groupby("layer")["frechet_dist"].mean()
    #dg_std = dg.groupby("layer")["frechet_dist"].std()
    dg_meanR = dg[dg['L1'] == 'ru'].groupby("layer")["frechet_dist"].mean()
    print(dg_meanR)
    dg_stdR = dg[dg['L1'] == 'ru'].groupby("layer")["frechet_dist"].std()
    dg_meanF = dg[dg['L1'] == 'fr'].groupby("layer")["frechet_dist"].mean()
    dg_stdF = dg[dg['L1'] == 'fr'].groupby("layer")["frechet_dist"].std()
    dg_meanV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["frechet_dist"].mean()
    dg_stdV = dg[dg['L1'] == 'fr vs ru'].groupby("layer")["frechet_dist"].std()
    fig = plt.figure()
    plt.xlabel('Layer')
    plt.ylabel('Frechet distance')
    plt.plot(dg_meanR.index, dg_meanR.values, "x-",label="ru speakers only")
    plt.fill_between(dg_meanR.index, dg_meanR.values - dg_stdR.values, dg_meanR.values + dg_stdR.values, alpha=0.3)
    plt.plot(dg_meanF.index, dg_meanF.values, "+-",label="fr speakers only")
    plt.fill_between(dg_meanF.index, dg_meanF.values - dg_stdF.values, dg_meanF.values + dg_stdF.values, alpha=0.3)
    plt.plot(dg_meanV.index, dg_meanV.values, "3-",label="speakers w. different L1")
    plt.fill_between(dg_meanV.index, dg_meanV.values - dg_stdV.values, dg_meanV.values + dg_stdV.values, alpha=0.3)
    plt.legend()
    plt.title('Vector Path to Audio Path Frechet distance')
    fig.figure.savefig(f"{args.target_word}_{suffnorm}_L1_effect_FRECHET_8SPK.pdf", bbox_inches='tight')
    plt.close()













