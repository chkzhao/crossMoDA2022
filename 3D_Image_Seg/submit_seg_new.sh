#!/bin/bash

# account=bio170034p
partition=GPU-shared
num_nodes=1
gpu=gpu:v100-32:1
wall_time=48:00:00

world_size=4
ranks=(0 1 2 3)
# world_size=1
# ranks=0
#
job_name=cross_moda_seg

for rank in "${ranks[@]}"
do
            RUN_CMD="
python3 VS_transfer.py --results_folder_name results/train_error_log.txt \
--dataset T2 --data_root ../../dataset/training/source_training_51/ --split ./params/split_crossMoDA2022_red.csv \
--train_batch_size 2 --cache_rate 1.0 --initial_learning_rate 1e-4 --model UNet2d5_spvPA --zoom_model UNet2d5_spvPA  \
--intensity 1800 --sync_bn --seg_only --weighted_crop --direction AtoB --patchD_size 384 \
--master_port 65521 --world_size $world_size --rank $rank"
            sbatch -p $partition -N $num_nodes --gres=$gpu -t $wall_time --job-name=$job_name ./submit_job.sh $RUN_CMD
done