# Intro : Un répertoire de dev

Ce répertoire ne remplace pas le répertoire *representations_analysis* mais est un exemple d'impléménetation pratique des
Il est plus près des données et sert à tester les développements de GW tout en étant qu plus proche des données.

# Quelques exemples de commande pour générer les fichiers qui seviront aux comparaisons

## D'abord, les données sont préparées, c'est à dire qu'elles sont extraites sous la forme d'un gros dataframe non aligné (un dataframe par couche)
>python prepare_data_mult_lay_by_lay.py
(stockage : ~/xxx_XP/05_mesure_distance/RUFR_res/preprocessed_corpora)

(ou                                                                                     )
(                                                                                       )
(>python prepare_data_selec_spk.py                                                      )
(                                                                                       )
(suivant qu'on veut ou non filtrer et ne garder qu'un speaker (utile en phases de dev)  )

## Ensuite, l'alignement de phrases se fait avec toutes les entrées données en dur dans le programme, et donc il suffit de lancer la commande :

>python align_sentences.py

Ce programme fournit en sortie un fichier *_aligned.pkl qui aligne en se basant sur les mots clés entrés dans la tire words des données d'entrée (annotations praat). Cette première version est à comprendre dans le cadre très rudimentaire des phrases cadre où le mot clé ne peut être qu'à un seul endroit. Cela signifie que dans les situations où le mot-clé se répète dans la phrase, il est plus que possible que l'alignement disfonctionne.
Par exemple, dans le cas où le mot-clé est cache-cache, si la tire "words" liste deux intervalles successives contenant chacune le mot "cache", alors il y a un risque de sur-alignement suivant les oprions choisie dans les fonctions *merge* et *explode*. Et donc pour cela il faut l'éviter, mais ce sera dans un second exemple **(à suivre)**.

## Enfin, le programme de calcul de distances.
Version d'avant aout 2024 :
>python compute_distances.py --input_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora/aligned --repr_column layer_24 --output cachecache_allspk_nonormalization.pkl

Version d'après aout 2024
>python compute_distances.py --normalize --input_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora/aligned --repr_column layer_24 --output_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/tulle --filt_W tulle

Version d'après aout 2024 sans les redondances (= moins les symétries)
>python compute_distances_noduplicates.py --normalize --input_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora/aligned --repr_column layer_24 --output_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/tulle --filt_W tulle

Version Finale (= moins les symétries, mais rajout des comparaisons A-A)
>python compute_distances_nopermut_only.py --normalize --input_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora/aligned --repr_column layer_24 --output_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/lay_blabla_tulle --filt_W tulle

Version qui focus sur le 3
>python compute_distances_mot_trois.py --normalize --input_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/preprocessed_corpora/aligned --repr_column layer_24 --output_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/tulle --filt_W1 tulle filt_W2 juxtaposer

# Pour les post-traitements

## Qualitatifs

le fichier make_plot.py (qui correspond à la troisième version, créée à la livraison de la version avec alignement de compute_distance.py) permet de tracer des premiers plots qui doivent nous éclairer sur lastructure des données de sortie. C'est normal si c'est un peu fouillis.

Commande en version été 2024 (obsolète):
>python make_plot.py --work_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/cachecache

Commande en version automne 2024 :
>python make_plot.py --work_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword --work_file hier-layer_24-allspk-NORM.pkl --line_num 1

Commande en version batch :
>python make_plot_batch.py --work_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/tulle_NORM --work_file tulle-layer_24-allspk-NORM.pkl

Commande pour regrouper en un seul graphique en intra-locuteur, les comparaisons sur mot cible, avec en ligne 1 les plus proches répétitions (dans le temps) et en dernière ligne les plus éloignées :
>python combine_plots_closest_first.py --work_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/tulle_NORM
(pas d'argument, c'est normal car on fait tous les locuteurs)

## Quantitatifs
Composition du nouveau tableau effectuant à la fois DTW vectoriel et audio :
>python make_table_with_DTW.py --work_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/lay_16_garage_NORM --work_file garage-layer_16-allspk-NORM.pkl

Obtention des graphes relatifs aux données macro couch par couche et aux profils de distance audio-vecto normalisées :
>python make_SSanova.py --root_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword --target_word juxtaposer --out_dir /home/mfily/Documents/NOANOA_locallyowned_texfiles/COLING_2025 --normalized

Obtention des boxplots données macro à couche fixe :
>python sensitivities.py --input_file /home/mfily/Documents/NOANOA_locallyowned_texfiles/COLING_2025/result_garage-layer_24-NORM.pkl --layer 24

multi-fichier :
>python sensitivities_NORM_UNNO_NRSP_mult_entries.py --input_file1 result_garage-NORM.pkl --input_file2 result_sérieux-NORM.pkl --input_file3 result_juxtaposer-NORM.pkl --input_file4 result_tulle-NORM.pkl --input_file5 result_hier-NORM.pkl --input_file6 result_tsarine-NORM.pkl --layer 20

## Traitements graphiques
>python make_db_DTW_graph_chose_x_y.py --work_dir /home/mfily/Documents/diagnoSTIC_XP/05_mesure_distance/RUFR_res/distance_mtx_byword/lay_16_garage_NORM --work_file garage-layer_16-allspk-NORM.pkl --fich_x FRcorp35 --fich_y FRcorp35 --locut_x AR --locut_y SH --out_dir /home/mfily/Documents/NOANOA_locallyowned_texfiles/COLING_2025


