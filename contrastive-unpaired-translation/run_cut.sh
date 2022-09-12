#! /bin/bash

source /ocean/projects/asc170022p/findlamp/miniconda3/etc/profile.d/conda.sh
conda activate base
  
python train.py --dataroot ../../../dataset/training/ \
--CUT_mode CUT --load_size 448 --crop_size 448 \
--batch_size 2 --num_threads 4 \
--direction AtoB \
--master_port 65535 --world_size 1 --rank 0 --gpu_ids 0,1,2,3,4,5