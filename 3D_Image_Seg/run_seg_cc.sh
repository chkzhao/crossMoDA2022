#! /bin/bash

source /ocean/projects/asc170022p/findlamp/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 VS_transfer_cc.py --results_folder_name results/train_error_log.txt \
--dataset T2 --data_root ../../dataset/training/source_training/ \
--train_batch_size 1 --cache_rate 1.0 --initial_learning_rate 1e-4 --model UNet2d5_spvPA \
--intensity 4100 --seg_only --weighted_crop --direction AtoB --patchD_size 384 \
--master_port 65530 --world_size 1 --rank 0
