#! /bin/bash

python train.py --dataroot ../data/cityscapes/ --model minimum_gan --lambda_minimum 0.01 --load_size 128 --crop_size 128 --batch_size 8 --num_threads 32 --pool_size 50 --theta_mix --netG stylegan2 --netD patch
