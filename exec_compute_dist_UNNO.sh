#!/bin/bash
for ((i = 0 ; i < 25 ; i++ ));
do
python compute_distances_nopermut_only.py --input_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora/lay_${i}_aligned --repr_column layer_${i} --output_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/lay_${i}_$1_UNNO --filt_W $1;
done
