#!/bin/bash

account=bio170034p
partition=BatComputer
num_nodes=1
gpu=gpu:rtx6000:1
wall_time=48:00:00

world_size=4
ranks=(0 1 2 3)
#
job_name=cross_moda_trans

for rank in "${ranks[@]}"
do
            RUN_CMD="
python train.py --dataroot ../../../dataset/training/ --continue_train --epoch_count 53 \
--CUT_mode CUT --load_size 448 --crop_size 440 \
--batch_size 4 --num_threads 32 \
--direction AtoB \
--master_port 65535 --world_size $world_size --rank $rank"
            sbatch -A $account -p $partition -N $num_nodes --gres=$gpu -t $wall_time --job-name=$job_name ./submit_job.sh $RUN_CMD
done
