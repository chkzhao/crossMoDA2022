#! /bin/bash

source /ocean/projects/asc170022p/findlamp/miniconda3/etc/profile.d/conda.sh
conda activate base

python test.py --dataroot ../../../dataset/training \
--CUT_mode CUT --load_size 448 --crop_size 440 \
--batch_size 4 --num_threads 32 \
--direction AtoB --eval --phase train --epoch 72 \
--master_port 65525 --world_size 1 --rank 0