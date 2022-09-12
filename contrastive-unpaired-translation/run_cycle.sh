#! /bin/bash

python train.py --dataroot ../data/cityscapes/ --model cycle_gan --load_size 128 --crop_size 128 --batch_size 8 --num_threads 32 --pool_size 0  --netG stylegan2 --netD patch
