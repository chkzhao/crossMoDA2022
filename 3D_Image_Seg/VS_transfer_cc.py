#!/usr/bin/env python
# coding: utf-8

import argparse
import monai
from params.networks.nets.dis import FCDiscriminator
import torch
from params.VS_params_cc import VSparams
import os
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

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
p = VSparams(parser)

# initialize parameters

# create folders
p.create_results_folders()

if p.rank==0:
    master_addr = socket.gethostbyname(socket.gethostname())
    master_file = open(p.master_port + '_master_file.txt', 'w')
    master_file.write(master_addr)
    master_file.close()
    setup(p.rank, p.world_size, master_addr, p.master_port)
else:
    while not os.path.exists(p.master_port + '_master_file.txt'):
        pass
    time.sleep(1)
    master_file = open(p.master_port + '_master_file.txt', 'r')
    master_addr = master_file.readlines()[0]
    master_file.close()
    setup(p.rank, p.world_size, master_addr, p.master_port)

# set up logger
logger = p.set_up_logger("training_log.txt")

p.log_parameters()

# load paths to data sets
all_files = p.load_T1_or_T2_data()

val_files = []
train_files = []

val_index = np.ndarray.tolist(np.random.choice(len(all_files),10,replace=False))

for i in range(len(all_files)):
    if i in val_index:
        val_files.append(all_files[i])
    else:
        train_files.append(all_files[i])

# # load paths to data sets
# train_files, val_files, test_files = p.load_T1_or_T2_data()
#
# val_files = []
#
# index = np.load('../data_o/data_preprocess_partial/index.npy')
#
# val_path = '../data_o/data_preprocess_partial/'
# for i in index:
#     if os.path.exists(os.path.join(val_path, 'crossmoda_'+str(i)+'_ceT1.nii.gz')):
#         if p.dataset == 'T1':
#             val_files.append({"image": os.patha.join(val_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz'),
#                               "label": os.path.join(val_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
#         elif p.dataset == 'T2':
#             val_files.append({"image": os.path.join(val_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                               "label": os.path.join(val_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
#         elif p.dataset == 'both':
#             val_files.append({"imageT1": os.path.join(val_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                               "imageT2": os.path.join(val_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                               "label": os.path.join(val_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})

# define the transforms
train_transforms, train_target_transforms, val_transforms, test_transforms = p.get_transforms()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=0)

# check transforms
p.check_transforms_on_first_validation_image_and_label(all_files, train_transforms)

# cache and load data
train_loader = p.cache_transformed_train_data(all_files, train_transforms)
val_loader = p.cache_transformed_val_data(val_files, val_transforms)

target_train_path = '../../dataset/training/target_training'
img_lists = os.listdir(target_train_path)

target_train_files = []
# for i in range(106,211):
#         if p.dataset == 'T1':
#             target_train_files.append(
#                 {"image": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz')})
#         elif p.dataset == 'T2':
#             target_train_files.append(
#                 {"image": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})
#         elif p.dataset == 'both':
#             target_train_files.append(
#                 {"imageT1": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                  "imageT2": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})
#         elif p.dataset == 'both_h':
#             target_train_files.append(
#                 {"imageT1": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                  "imageT2": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                  "imageH": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hist.nii.gz')})
#         else:
#             target_train_files.append(
#                 {"image": os.path.join(target_train_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})

for img_name in img_lists:
    target_train_files.append(
        {"image": os.path.join(target_train_path, img_name)})

target_train_loader = p.cache_transformed_train_data(target_train_files, train_target_transforms)

nets = p.set_and_get_model()
if p.args.transfer_seg and p.args.load_dict:
    nets = p.load_trained_state_of_model(nets)
loss_function = p.set_and_get_loss_function()

# run training algorithm
p.run_training_algorithm(nets, loss_function, train_loader, target_train_loader, val_loader)
cleanup()

# plot loss and mean dice
# p.plot_loss_curve_and_mean_dice(epoch_loss_values, metric_values)
