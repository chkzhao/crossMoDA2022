#! /bin/bash

python train.py --dataroot ../data/horse2zebra/ --model minimum_gan --lambda_minimum 0.0 --load_size 286 --crop_size 256 --batch_size 8 --num_threads 32 --pool_size 50 --theta_mix --netG stylegan2 --netD patch
