#!/bin/bash
for ((i = 0 ; i < 25 ; i++ ));
do
#python make_table_with_DTW.py --work_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/lay_${i}_$1_NORM --work_file $1-layer_${i}-allspk-NORM.pkl
#python make_table_with_DTW.py --work_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/lay_${i}_$1_UNNO --work_file $1-layer_${i}-allspk-UNNO.pkl
python make_table_with_DTW.py --work_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/lay_${i}_$1_NSPK --work_file $1-layer_${i}-allspk-NSPK.pkl
done
