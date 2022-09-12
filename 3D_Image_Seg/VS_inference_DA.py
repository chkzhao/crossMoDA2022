#!/usr/bin/env python
# coding: utf-8

import argparse
import monai
import torch
from params.VSparams import VSparams
import os

torch.backends.cudnn.benchmark = True

monai.config.print_config()

# read parsed arguments
parser = argparse.ArgumentParser(description="Train the model")

# initialize parameters
p = VSparams(parser)

p.results_folder_path = os.path.join("../results", p.dataset, p.model,
                                             str(p.args.warm_seg) + '_' + ('DA_' if p.args.DA else '') + ('w_crop_' if p.args.weighted_crop else '') +
                                               'dist_' + 'INS_all_' + p.results_folder_name)

p.logs_path = os.path.join(p.results_folder_path, "logs")
p.model_path = os.path.join(p.results_folder_path, "model")
p.figures_path = os.path.join(p.results_folder_path, "figures")

# create folders
p.create_results_folders()
# set up logger
logger = p.set_up_logger("test_log.txt")

# log parameters
p.log_parameters()

# load paths to data sets
test_files = []

test_path = '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/target_validation'

for i in range(211,243):
    if p.dataset == 'T1':
        test_files.append({"image": os.path.join(test_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz')})
    elif p.dataset == 'T2':
        test_files.append({"image": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})
    elif p.dataset == 'both':
        test_files.append({"imageT1": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                           "imageT2": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})
    elif p.dataset == 'both_h':
        test_files.append({"imageT1": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                           "imageT2": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                           "imageH": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})
    else:
        test_files.append(
            {"image": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz')})


print(test_files)


# define the transforms
train_transforms,_, val_transforms, test_transforms = p.get_transforms()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=0)

# cache and load validation data
test_loader = p.cache_transformed_test_data(test_files, test_transforms)

# create UNet
nets = p.set_and_get_model()

# load the trained state of the model
model = p.load_trained_state_of_model(nets)

# run inference and create figures in figures folder
p.run_inference(model, test_loader)
