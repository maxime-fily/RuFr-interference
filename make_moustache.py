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

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", required=True, type=Path)
parser.add_argument("--work_file", required=True, type=Path)
#parser.add_argument("--legend", required=False, type=str)
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
        print(row['ident'])
        return str(sorted([row[cx],row[cy]])[0]+" vs "+sorted([row[cx],row[cy]])[1])

if __name__ == "__main__":
#    C1_x=args.colhue_x
#    C1_y=args.colhue_y
#    legd=args.legend
    os.chdir(args.work_dir)
    df=pd.read_pickle(args.work_file)
    print(list(df))

    #df['same_speaker'] = df['spk_x'] == df['spk_y']
    #sns.violinplot(x='same_speaker', y='dist_N', data=df)
    #plt.show()

    #filter dataframe so that it does not count the same-same cos dist.
    #This is important to avoid a bias towards the low values of distance, and it has no interest to keep them anyway.
    df = df[~((df['spk_x'] == df['spk_y']) & (df['file_x'] == df['file_y']))]
    print(len(df))
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
    #distinguer selon la L1 du locuteur également
    db_merged["L1"] = db_merged.apply(categorize, axis=1, cx='L1_x', cy='L1_y')
    listCOMPL=[]
    dadd=pd.DataFrame()
    for index, couple in df.iterrows():
        LD=couple["DTW"].costMatrix[-1, -1]/(db_merged["m"]*db_merged["n"])
        listCOMPL.append([LD])
    dadd = pd.DataFrame(listCOMPL)
    dadd.columns = ["Ncost_LD"]
    db_merged=pd.concat([db_merged,dadd], axis=0)
    plot = sns.boxplot(x='same_speaker', y='dist_A', hue="L1", data=db_merged)
    plot.set_ylabel('Min. Path Dist (Abs.)')
    plot.set_title("Effect of speaker ID (same or different)")
    plot.figure.savefig(f"boxplot_L1_effect.pdf",bbox_inches='tight')
    plt.clf()
    allbutsamespk=db_merged[db_merged['same_speaker'] == False].copy()
    print(len(allbutsamespk))
    allbutsamespk['same_sex'] = allbutsamespk['sex_x'] == allbutsamespk['sex_y']
    allbutsamespk["AGAB"] = db_merged.apply(categorize, axis=1, cx='sex_x', cy='sex_y')
    plou = sns.boxplot(x='same_sex', y='dist_A', hue="AGAB", data=allbutsamespk)
    plou.set_ylabel('Min. Path Dist (Abs.)')
    plou.set_title("Effect of L1 (ru or fr)")
    plou.figure.savefig(f"boxplot_gender_effect.pdf",bbox_inches='tight')
    plt.clf()
    allbutsamespk['same_lng'] = allbutsamespk['L1_x'] == allbutsamespk['L1_y']
    allbutsamespk["level"] = db_merged.apply(categorize, axis=1, cx='level_fr_x', cy='level_fr_y')
    plov = plt.figure(figsize=(20, 10))
    plov = sns.boxplot(x='same_lng', y='dist_A', hue="level", data=allbutsamespk, palette='Paired')
    plov.set_ylabel('Min. Path Dist (Abs.)')
    plov.set_title("Effect of CEFR level")
    plov.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plov.figure.savefig(f"boxplot_level_effect.pdf")
    plt.clf()
