import SimpleITK as sitk
import os
import numpy as np

import torch.multiprocessing as mp
mp = mp.get_context('spawn')

img_path =  '/ocean/projects/asc170022p/yanwuxu/DA/results/T12T2_resample/cut0.001_4_448_BtoA_resnet_9blocks_basic/train_latest/images/fake_B'

file_lists = os.listdir(img_path)
file_lists.sort()
print(file_lists)
nii_img = []
file_num = -1
for file in file_lists:

    file_index = int((file.replace('.npy','').split('_'))[0])

    if file_num != file_index:
        if len(nii_img) != 0:
            nii_img = np.stack(nii_img, axis=0)
            print(nii_img.shape)
            nii_img = sitk.GetImageFromArray(nii_img)
            source_img = sitk.ReadImage(os.path.join(
                '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/crossmoda_training/source_training',
                'crossmoda_' + str(file_num) + '_ceT1.nii.gz'))
            nii_img.CopyInformation(source_img)
            print(os.path.join(
                '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/crossmoda_training/source_training',
                'crossmoda_' + str(file_num) + '_hrT2.nii.gz'))
            sitk.WriteImage(nii_img, os.path.join(
                '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/crossmoda_training/source_training',
                'crossmoda_' + str(file_num) + '_hrT2.nii.gz'))
            nii_img = []
        file_num = file_index


    img = np.load(os.path.join(img_path, file))
    img = np.squeeze((img+1.0)*4100.0/2.0).astype(np.int16)
    nii_img.append(img)

nii_img = np.stack(nii_img, axis=0)
print(nii_img.shape)
nii_img = sitk.GetImageFromArray(nii_img)
source_img = sitk.ReadImage(os.path.join(
                '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/crossmoda_training/source_training',
                'crossmoda_' + str(file_num) + '_ceT1.nii.gz'))
nii_img.CopyInformation(source_img)
sitk.WriteImage(nii_img, os.path.join(
                '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/crossmoda_training/source_training',
                'crossmoda_' + str(file_num) + '_hrT2.nii.gz'))
