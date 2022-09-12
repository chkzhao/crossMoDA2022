#!/usr/bin/env python
# coding: utf-8

import argparse
import monai
import torch
from params.VSparams_T import VSparams
import os

import time
import socket
import torch.distributed as dist

torch.backends.cudnn.benchmark = True

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
logger = p.set_up_logger("test_log.txt")

# log parameters
p.log_parameters()

# load paths to data sets
test_files = []

test_path = '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/target_validation'
# test_path = '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/crossmoda_training/target_training'

for i in range(211,243):
    if p.dataset == 'T1':
        test_files.append({"image": os.path.join(test_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz'),
                           "label": os.path.join(test_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
    elif p.dataset == 'T2':
        test_files.append({"image": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                           "label": os.path.join(test_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
    elif p.dataset == 'both':
        test_files.append({"imageT1": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                           "imageT2": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                           "label": os.path.join(test_path, 'crossmoda_' + str(i) + '_Label.nii.gz'), })
    elif p.dataset == 'both_h':
        test_files.append({"imageT1": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                           "imageT2": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                           "imageH": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
                           "label": os.path.join(test_path, 'crossmoda_' + str(i) + '_Label.nii.gz'), })
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
model, zoom_model, G, _, _, _, _ = nets

# run inference and create figures in figures folder
p.run_inference(model,zoom_model, test_loader)

# test_path = '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/crossmoda_training/source_training'

# test_files = []
# for i in range(1,106):
#     if p.dataset == 'T1':
#         test_files.append({"image": os.path.join(test_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz'),
#                            "label": os.path.join(test_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
#     elif p.dataset == 'T2':
#         test_files.append({"image": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                            "label": os.path.join(test_path, 'crossmoda_' + str(i) + '_Label.nii.gz')})
#     elif p.dataset == 'both':
#         test_files.append({"imageT1": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                            "imageT2": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                            "label": os.path.join(test_path, 'crossmoda_' + str(i) + '_Label.nii.gz'), })
#     elif p.dataset == 'both_h':
#         test_files.append({"imageT1": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                            "imageT2": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                            "imageH": os.path.join(test_path, 'crossmoda_' + str(i) + '_hrT2.nii.gz'),
#                            "label": os.path.join(test_path, 'crossmoda_' + str(i) + '_Label.nii.gz'), })
#     else:
#         test_files.append(
#             {"image": os.path.join(test_path, 'crossmoda_' + str(i) + '_ceT1.nii.gz')})
#
# print(test_files)
# test_loader = p.cache_transformed_test_data(test_files, test_transforms)
# p.run_transfer(G, test_loader)
cleanup()
