#! /bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate crossMoDA
python3 VS_transfer.py --results_folder_name results/train_error_log.txt \
--dataset transfer --data_root ../data_resampled/crossmoda_training/source_training --load_dict \
--train_batch_size 1 --cache_rate 1.0 --initial_learning_rate 1e-4 --model UNet2d5_spvPA \
--intensity 4100 --sync_bn --no_seg --direction AtoB --patchD_size 384 --warm_seg 202 --warm_transfer 202 \
--master_port 69280 --world_size 4 --rank 3
