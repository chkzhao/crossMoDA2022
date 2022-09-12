#!/usr/bin/env python
# coding: utf-8

import argparse
import monai

from params.VSparams import VSparams
import numpy as np
import os

monai.config.print_config()

# read parsed arguments
parser = argparse.ArgumentParser(description="Train the model")

# initialize parameters
p = VSparams(parser)

# create folders
p.create_results_folders()

# set up logger
logger = p.set_up_logger("training_log.txt")

# log parameters
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

# index = np.load('../data_o/t1_to_t2/index.npy')
#
# val_path = '../data_o/t1_to_t2/'
# for i in index:
#     if os.path.exists(os.path.join(val_path, 'crossmoda_'+str(i)+'_ceT1.nii.gz')):
#         if p.dataset == 'T1':
#             val_files.append({"image": os.path.join(val_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz'),
#                               "label": os.path.join(val_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
#         elif p.dataset == 'T2':
#             val_files.append({"image": os.path.join(val_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                               "label": os.path.join(val_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
#         elif p.dataset == 'both_sep':
#             val_files.append({"image": os.path.join(val_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                               "label": os.path.join(val_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
#         elif p.dataset == 'both':
#             val_files.append({"imageT1": os.path.join(val_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz'),
#                               "imageT2": os.path.join(val_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                               "label": os.path.join(val_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})

# define the transforms
train_transforms ,_ , val_transforms, test_transforms = p.get_transforms()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=0)

# check transforms
p.check_transforms_on_first_validation_image_and_label(val_files, val_transforms)

# cache and load data
train_loader = p.cache_transformed_train_data(all_files, train_transforms)
val_loader = p.cache_transformed_val_data(val_files, val_transforms)

# create UNet, DiceLoss and Adam optimizer
model = p.set_and_get_model()
loss_function = p.set_and_get_loss_function()
optimizer = p.set_and_get_optimizer(model)

# run training algorithm
epoch_loss_values, metric_values = p.run_training_algorithm(model, loss_function, optimizer, train_loader, val_loader)

# plot loss and mean dice
p.plot_loss_curve_and_mean_dice(epoch_loss_values, metric_values)
