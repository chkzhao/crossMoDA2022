#! /bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate crossMoDA
python3 VS_DA.py --results_folder_name results/train_error_log.txt --dataset da --DA \
--data_root ../registered_data_t1_2_t2/crossmoda_training/source_training \
--train_batch_size 1 --cache_rate 1.0 --initial_learning_rate 1e-4 --model UNet2d5_spvPA --weighted_crop \
--master_port 69285 --world_size 10 --rank 9
