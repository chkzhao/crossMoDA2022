#! /bin/bash
  
source /jet/home/yanwuxu/miniconda3/etc/profile.d/conda.sh
conda activate crossMoDA
python3 VS_inference.py --results_folder_name results/train_error_log.txt --dataset both_h --data_root ../registered_data_t1_2_t2/crossmoda_training/source_training --train_batch_size 1 --cache_rate 0.0 --initial_learning_rate 1e-4 --model UNet2d5_spvPA
