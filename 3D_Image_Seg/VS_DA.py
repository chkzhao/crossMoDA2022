#!/usr/bin/env python
# coding: utf-8

import argparse
import monai
from params.networks.nets.dis import FCDiscriminator
import torch
from params.VSparams import VSparams
import os
import numpy as np
import torch.distributed as dist

import socket

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

monai.config.print_config()

# read parsed arguments
parser = argparse.ArgumentParser(description="Train the model")

# initialize parameters
p = VSparams(parser)

if p.args.rank == 0:
    master_addr = socket.gethostbyname(socket.gethostname())
    master_file = open(p.args.master_port + '_master_file.txt', 'w')
    master_file.writelines(master_addr)
    master_file.close()
    setup(p.args.rank, p.args.world_size, master_addr, p.args.master_port)
else:
    while not os.path.exists(p.args.master_port + '_master_file.txt'):
        pass
    master_file = open(p.args.master_port + '_master_file.txt', 'r')
    master_addr = master_file.read()
    print(master_addr)
    master_file.close()
    setup(p.args.rank, p.args.world_size, master_addr, p.args.master_port)

# set up logger

logger = p.set_up_logger('training_log' + '.txt')
p.log_parameters()

# load paths to data sets
source_train_path = '../data_resampled/crossmoda_training/source_training'
source_train_files = []
for i in range(1,106):
        if p.dataset == 'T1':
            source_train_files.append(
                {"image": os.path.join(source_train_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz')})
        elif p.dataset == 'T2':
            source_train_files.append(
                {"image": os.path.join(source_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})
        elif p.dataset == 'both':
            source_train_files.append(
                {"imageT1": os.path.join(source_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                 "imageT2": os.path.join(source_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})
        elif p.dataset == 'both_h':
            source_train_files.append(
                {"imageT1": os.path.join(source_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                 "imageT2": os.path.join(source_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                 "imageH": os.path.join(source_train_path, 'crossmoda_' + str(i) + '_hist.nii.gz')})
        else:
            source_train_files.append(
                {"image": os.path.join(source_train_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz'), 'label': os.path.join(source_train_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
            img_t2path = '../results/transfer/UNet2d5_spvPA/60_100_4100.0Transfer_Seg_w_crop_dist_noload_DA_INS_all_results/train_error_log.txt/transfered_hrT2'
            source_train_files.append(
                {"image": os.path.join(img_t2path, 'crossmoda_' + str(i) + '_hrT2', 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                 'label': os.path.join(source_train_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})

val_files = []
train_files = []

val_index = np.ndarray.tolist(np.random.choice(len(source_train_files),10,replace=False))

for i in range(len(source_train_files)):
    if i in val_index:
        val_files.append(source_train_files[i])
    else:
        train_files.append(source_train_files[i])

# define the transforms
train_transforms, train_target_transforms, val_transforms, test_transforms = p.get_transforms()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=0)

# check transforms
p.check_transforms_on_first_validation_image_and_label(val_files, val_transforms)

# cache and load data
train_loader = p.cache_transformed_train_data(source_train_files, train_transforms)
val_loader = p.cache_transformed_val_data(val_files, val_transforms)

target_train_path = '../data_resampled/crossmoda_training/target_training'
target_train_files = []
for i in range(106,211):
    if i==120:
        pass
    else:
        if p.dataset == 'T1':
            target_train_files.append(
                {"image": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz')})
        elif p.dataset == 'T2':
            target_train_files.append(
                {"image": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})
        elif p.dataset == 'both':
            target_train_files.append(
                {"imageT1": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                 "imageT2": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})
        elif p.dataset == 'both_h':
            target_train_files.append(
                {"imageT1": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                 "imageT2": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                 "imageH": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hist.nii.gz')})
        else:
            label_name = os.path.join('../results/transfer/UNet2d5_spvPA/60_100_4100.0Transfer_Seg_w_crop_dist_noload_DA_INS_all_results/train_error_log.txt/inferred_segmentations_nifti',
                                      'crossmoda_' + str(i) + '_Label', 'crossmoda_' + str(i) + '_Label' + '.nii.gz')
            target_train_files.append(
                {"image": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),'label': label_name})

target_train_loader = p.cache_transformed_train_data(target_train_files, train_target_transforms)

nets = p.set_and_get_model()
loss_function = p.set_and_get_loss_function()
optimizers = p.set_and_get_optimizer(nets)


# run training algorithm
p.run_training_algorithm_DA(nets, loss_function, optimizers, train_loader, target_train_loader, val_loader)
cleanup()

# plot loss and mean dice
# p.plot_loss_curve_and_mean_dice(epoch_loss_values, metric_values)
