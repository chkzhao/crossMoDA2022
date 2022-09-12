#!/usr/bin/env python
# coding: utf-8
import monai
import torch
import os
torch.backends.cudnn.benchmark = True
from params.networks.nets.unet2d5_spvPA import UNet2d5_spvPA_zoom, UNet2d5_spvPA
from monai.networks.layers import Norm
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
import numpy as np
from params.crop_bbox import crop_seg
import nibabel as nib
from monai.data import NiftiSaver

from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ThresholdIntensityd,
    NormalizeIntensityd,
    Orientationd,
    ToTensord
)


def set_and_get_model():
    input_dim = 1

    model = UNet2d5_spvPA(
            zoom_in=True,
            dimensions=3,
            in_channels=input_dim,
            out_channels=3,
            channels=(16, 32, 48, 64, 80, 96),
            strides=(
                (2, 2, 1),
                (2, 2, 1),
                (2, 2, 2),
                (1, 1, 1),
                (1, 1, 1),
            ),
            kernel_sizes=(
                (3, 3, 1),
                (3, 3, 1),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            sample_kernel_sizes=(
                (3, 3, 1),
                (3, 3, 1),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.1,
            attention_module=True,
        ).cuda()

    model_dict = torch.load('/ocean/projects/asc170022p/yanwuxu/crossMoDA/results/both_sep/UNet2d5_spvPAUNet2d5_spvPA/202_202_4100.0Transfer_w_crop_dist_sync_bn_seg_only_noload_DA_INS_all_results/train_error_log.txt/model/125_epoch_model.pth')
    keys = list(model_dict.keys())
    for key in keys:
        model_dict[key.replace('module.','').replace('sub','submodule.')] = model_dict.pop(key)
    model.load_state_dict(model_dict)

    zoom_model = UNet2d5_spvPA_zoom(
            zoom_in= True,
            dimensions=3,
            in_channels=input_dim,
            out_channels=2,
            channels=(16, 32, 48, 64, 80, 96),
            strides=(
                (2, 2, 1),
                (2, 2, 1),
                (2, 2, 2),
                (1, 1, 1),
                (1, 1, 1),
            ),
            kernel_sizes=(
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            sample_kernel_sizes=(
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
                (3, 3, 3),
            ),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.1,
            attention_module=True,
        ).cuda()
    zoom_model_dict = torch.load(
        '/ocean/projects/asc170022p/yanwuxu/crossMoDA/results/both_sep/UNet2d5_spvPAUNet2d5_spvPA/202_202_4100.0Transfer_w_crop_dist_sync_bn_seg_only_noload_DA_INS_all_results/train_error_log.txt/model/125_epoch_zoom_model.pth')
    keys = list(zoom_model_dict.keys())
    for key in keys:
        zoom_model_dict[key.replace('module.', '').replace('sub', 'submodule.')] = zoom_model_dict.pop(key)
    zoom_model.load_state_dict(zoom_model_dict)
    return model, zoom_model

model, zoom_model = set_and_get_model()

input_dir = '/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/target_validation/'
path_img = os.path.join(input_dir,'crossmoda_{}_hrT2.nii.gz')
path_pred = '/output/crossmoda_{}_Label.nii.gz'

list_case = [k.split('_')[1] for k in os.listdir(input_dir)]

# load paths to data sets
test_files = []

for case in list_case:
    test_files.append(
            {"image": path_img.format(case)})
print(test_files)

intensity = 4100.0
test_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    ThresholdIntensityd(keys=["image"], threshold=0, above=True),
                    ThresholdIntensityd(keys=["image"], threshold=intensity, above=False),
                    NormalizeIntensityd(keys=["image"], subtrahend=intensity/2, divisor=intensity/2),
                    NormalizeIntensityd(keys=["image"]),
                    ToTensord(keys=["image"]),
                ]
            )

# cache and load validation data
def cache_transformed_test_data(test_files, test_transforms):
    test_ds = monai.data.CacheDataset(data=test_files, transform=test_transforms, cache_rate=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    return test_loader

test_loader = cache_transformed_test_data(test_files, test_transforms)

# create UNet
nets = set_and_get_model()
model, zoom_model = nets

results_folder_path = './resample_label'
if not os.path.exists(results_folder_path):
    os.mkdir(results_folder_path)

def run_inference( model, zoom_model, data_loader=None):

    for m in model.modules():
        if isinstance(m, torch.nn.SyncBatchNorm):
            m.track_running_stats = False

    for m in zoom_model.modules():
        if isinstance(m, torch.nn.SyncBatchNorm):
            m.track_running_stats = False

    model_segmentation = lambda *args, **kwargs: model(*args, **kwargs)[0]

    zoom_model_segmentation = lambda *args, **kwargs: zoom_model(*args, **kwargs)[0]

    with torch.no_grad():  # turns off PyTorch's auto grad for better performance
        for i, data_dir in enumerate(test_files):

            data = test_transforms(data_dir)
            inputs = data["image"].cuda().unsqueeze(dim=0)


            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size= [384, 384, 48],
                sw_batch_size=1,
                predictor=model_segmentation,
                overlap=0.5,
                sigma_scale=0.12,
                mode="gaussian"
            )

            # out_max, y_pred = outputs.max(dim=1, keepdim=True)
            # y_pred[(out_max<=0.95) & (y_pred != 2)] = 0

            y_pred = torch.argmax(outputs, dim=1, keepdim=True)

            seg_map = y_pred.squeeze() * 1
            seg_map[seg_map != 2] = 0
            seg_map[seg_map == 2] = 1

            bbox_lists = crop_seg(seg_map.cpu().numpy(), coc_size=[32, 32, 16], center=True)

            for bbox in bbox_lists:
                zoom_inputs = inputs[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
                zoom_y_pred = y_pred[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
                zoom_outputs = sliding_window_inference(
                    inputs=zoom_inputs,
                    roi_size=[32, 32, 16],
                    sw_batch_size=1,
                    predictor=zoom_model_segmentation,
                    mode="gaussian",
                )

                zoom_y_pred_sub = torch.argmax(zoom_outputs, dim=1, keepdim=True)

                zoom_y_pred[zoom_y_pred_sub == 1] = 2
                y_pred[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = zoom_y_pred

            nifti_data_matrix = np.squeeze(y_pred)[None, :]
            img_dict = 'image_meta_dict'

            data[img_dict]['filename_or_obj'] = os.path.join(data_dir['image'].replace('hrT2','Label'))
            data[img_dict]['affine'] = np.squeeze(data[img_dict]['affine'])
            data[img_dict]['original_affine'] = np.squeeze(data[img_dict]['original_affine'])

            print(os.path.join(results_folder_path, 'inferred_segmentations_nifti'))
            saver = NiftiSaver(output_dir=os.path.join(results_folder_path), output_postfix='')
            saver.save(nifti_data_matrix, meta_data=data[img_dict])
            print(data[img_dict]['filename_or_obj'], np.unique(nifti_data_matrix.cpu().numpy()))

run_inference(model,zoom_model)