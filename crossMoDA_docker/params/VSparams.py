import os
import logging
import numpy as np
from natsort import natsorted
from time import perf_counter
import glob
import csv
from time import strftime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from matplotlib import pyplot as plt
import monai
import os
import torch.nn.functional as F
import nibabel as nib
import torch.nn as nn
import time
from .networks.nets.transfer_net import PatchDiscriminator_3D
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ThresholdIntensityd,
    NormalizeIntensityd,
    SpatialPadd,
    RandFlipd,
    RandSpatialCropd,
    Orientationd,
    ToTensord,
    RandAdjustContrastd,
    RandAffined,
    RandRotated,
    RandZoomd,
    RandSpatialCropd,
    RandCropByLabelClassesd,
    Identityd
)

from monai.networks.layers import Norm
from monai.data import NiftiSaver
# from torchviz import make_dot
# import hiddenlayer as hl
from .networks.nets.unet2d5_spvPA import UNet2d5_spvPA
from .networks.nets.unet_assp import unet_assp
from .networks.nets.UNet_3Plus import UNet_3Plus
from .networks.nets.dis import FCDiscriminator
from .losses.dice_spvPA import Dice_spvPA, DiceLoss
from monai.inferers import sliding_window_inference
from .new_dataset import CacheDataset_v2

monai.config.print_config()

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def adjust_learning_rate(lr, optimizer,batch_size, data_num, epochs):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = lr * (
        1.0 - float(group['step']) * float(batch_size) / (data_num * float(epochs)))
    return


class VSparams:
    def __init__(self, parser):
        parser.add_argument("--debug", dest="debug", action="store_true", help="activate debugging mode")
        parser.set_defaults(debug=False)

        parser.add_argument("--split", type=str, default="./params/split_crossMoDA.csv", help="path to CSV file that defines"
                                                                                         " training, validation and"
                                                                                         " test datasets")
        parser.add_argument("--dataset", type=str, default="T2", help='(string) use "T1" or "T2" to select dataset')
        parser.add_argument("--train_batch_size", type=int, default=1, help="batch size of the forward pass")
        parser.add_argument("--cache_rate", type=float, default=0.0, help="batch size of the forward pass")
        parser.add_argument("--initial_learning_rate", type=float, default=0.0001, help="learning rate at first epoch")
        parser.add_argument("--intensity", type=float, default=3000, help="learning rate at first epoch")
        parser.add_argument(
            "--no_attention",
            dest="attention",
            action="store_false",
            help="disables the attention module in "
            "the network and the attention map "
            "weighting in the loss function",
        )
        parser.set_defaults(attention=True)
        parser.add_argument(
            "--no_hardness",
            dest="hardness",
            action="store_false",
            help="disables the hardness weighting in " "the loss function",
        )
        parser.set_defaults(hardness=True)
        parser.add_argument(
            "--results_folder_name", type=str, default="temp" + strftime("%Y%m%d%H%M%S"), help="name of results folder"
        )
        parser.add_argument(
            "--weighted_crop",
            action="store_true",
            help="weighted crop label",
        )

        parser.add_argument(
            "--data_root", type=str, default= '', help="name of results folder"
        )
        parser.add_argument(
            "--model", type=str, default='', help="name of model"
        )

        parser.add_argument(
            "--EMG", action='store_true', help='entropy minimization regularization'
        )

        parser.add_argument(
            "--DA", action='store_true', help='Domain Adversarial loss'
        )

        parser.add_argument(
            "--G_C", action='store_true', help='geometry consistency'
        )

        parser.add_argument("--warm_seg", type=int, default=60, help="learning rate at first epoch")
        parser.add_argument('--world_size', type=int, default=4, help='world size of distrbuted learning')
        parser.add_argument('--rank', type=int, default=0, help='rank of distrbuted learning')
        parser.add_argument('--master_port', type=str, default='69280', help='rank of distrbuted learning')

        args = parser.parse_args()

        self.args = args

        self.debug = args.debug
        self.dataset = args.dataset
        self.data_root = args.data_root
        self.split_csv = args.split
        self.cache_rate = args.cache_rate
        self.results_folder_name = args.results_folder_name
        self.EMG = args.EMG
        self.G_C = args.G_C
        slice_num = 48
        self.img_size = 384
        if self.debug:
            self.split_csv = "./params/split_debug.csv"
        self.pad_crop_shape = [self.img_size, self.img_size, slice_num]
        if self.debug:
            self.pad_crop_shape = [128, 128, slice_num]
        self.pad_crop_shape_test = [self.img_size, self.img_size, slice_num]
        if self.debug:
            self.pad_crop_shape_test = [128, 128, slice_num]
        self.num_workers = 16
        self.torch_device_arg = "cuda:0"
        self.train_batch_size = args.train_batch_size
        self.initial_learning_rate = args.initial_learning_rate
        self.epochs_with_const_lr = 100
        if self.debug:
            self.epochs_with_const_lr = 3
        self.lr_divisor = 2.0
        self.weight_decay = 1e-5
        self.num_epochs = 200
        # self.num_epochs = int(1.2*self.num_epochs/self.world_size)
        if self.debug:
            self.num_epochs = 10
        self.val_interval = 2  # determines how frequently validation is performed during training
        self.model = args.model  # 'unet_assp' #"UNet"#"UNet2d5_spvPA"#
        self.sliding_window_inferer_roi_size = [self.img_size, self.img_size, slice_num]
        if self.debug:
            self.sliding_window_inferer_roi_size = [128, 128, slice_num]
        self.attention = args.attention
        self.hardness = args.hardness
        self.export_inferred_segmentations = True

        # paths
        # self.results_folder_path = os.path.join("../results", self.dataset, self.model, args.results_folder_name) #self.data_root
        self.results_folder_path = self.results_folder_path = os.path.join("../results", self.dataset, self.model,
                                             str(self.args.warm_seg) + '_' + ('DA_' if self.args.DA else '') + ('w_crop_' if self.args.weighted_crop else '') +
                                               ('dist_' if self.args.world_size >= 2 else '') + 'INS_all_' + self.results_folder_name)
        if self.debug:
            self.results_folder_path = os.path.join(self.data_root, "results", "debug")
        self.logs_path = os.path.join(self.results_folder_path, "logs")
        self.model_path = os.path.join(self.results_folder_path, "model")
        self.figures_path = os.path.join(self.results_folder_path, "figures")
        self.create_results_folders()

        #
        self.device = torch.device(self.torch_device_arg)

    def create_results_folders(self):
        # create results folders for logs, figures and model
        if self.args.rank==0:
            if not os.path.exists(self.logs_path):
                print(self.logs_path)
                os.makedirs(self.logs_path, exist_ok=False)
                os.chmod(self.logs_path, 0o777)
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path, exist_ok=False)
                os.chmod(self.model_path, 0o777)
            if not os.path.exists(self.figures_path):
                os.makedirs(self.figures_path, exist_ok=False)
                os.chmod(self.figures_path, 0o777)

    def set_up_logger(self, log_file_name):
        # logging settings
        self.logger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(self.logs_path, log_file_name), mode="w")
        consoleHandler = logging.StreamHandler()
        self.logger.addHandler(fileHandler)
        self.logger.addHandler(consoleHandler)
        formatter = logging.Formatter("%(asctime)s %(levelname)s        %(message)s")
        fileHandler.setFormatter(formatter)
        consoleHandler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.info("Created " + log_file_name)
        return self.logger

    def log_parameters(self):
        logger = self.logger
        # write all parameters to log
        logger.info("-" * 10)
        logger.info("Parameters: ")
        logger.info("dataset =                          {}".format(self.dataset))
        logger.info("data_root =                        {}".format(self.data_root))
        logger.info("split_csv =                        {}".format(self.split_csv))
        logger.info("pad_crop_shape =                   {}".format(self.pad_crop_shape))
        logger.info("pad_crop_shape_test =              {}".format(self.pad_crop_shape_test))
        logger.info("num_workers =                      {}".format(self.num_workers))
        logger.info("torch_device_arg =                 {}".format(self.torch_device_arg))
        logger.info("train_batch_size =                 {}".format(self.train_batch_size))
        logger.info("initial_learning_rate =            {}".format(self.initial_learning_rate))
        logger.info("epochs_with_const_lr =             {}".format(self.epochs_with_const_lr))
        logger.info("lr_divisor =                       {}".format(self.lr_divisor))
        logger.info("weight_decay =                     {}".format(self.weight_decay))
        logger.info("num_epochs =                       {}".format(self.num_epochs))
        logger.info("val_interval =                     {}".format(self.val_interval))
        logger.info("model =                            {}".format(self.model))
        logger.info("sliding_window_inferer_roi_size =  {}".format(self.sliding_window_inferer_roi_size))

        logger.info("attention =                        {}".format(self.attention))
        logger.info("hardness =                         {}".format(self.hardness))

        logger.info("results_folder_path =              {}".format(self.results_folder_path))
        logger.info("export_inferred_segmentations =    {}".format(self.export_inferred_segmentations))
        logger.info("-" * 10)

    def load_T1_or_T2_data(self):
        logger = self.logger

        train_files, val_files, test_files = [], [], []

        with open(self.split_csv) as csvfile:
            csvReader = csv.reader(csvfile)
            for row in csvReader:
                if self.dataset == "T1":
                    image_name = os.path.join(self.data_root, row[0]+ '_ceT1.nii.gz')
                    label_name = os.path.join(self.data_root, row[0] + '_Label.nii.gz')
                elif self.dataset == "T2":
                    image_name = os.path.join(self.data_root,  row[0]+ '_hrT2.nii.gz')
                    label_name = os.path.join(self.data_root,  row[0]+ '_Label.nii.gz')
                elif self.dataset == "both":
                    image_name_T1 = os.path.join(self.data_root, row[0]+ '_ceT1.nii.gz')
                    image_name_T2 = os.path.join(self.data_root,  row[0]+ '_hrT2.nii.gz')
                    label_name = os.path.join(self.data_root, row[0] + '_Label.nii.gz')
                elif self.dataset == "both_h":
                    image_name_T1 = os.path.join(self.data_root, row[0]+ '_ceT1.nii.gz')
                    image_name_T2 = os.path.join(self.data_root,  row[0]+ '_hrT2.nii.gz')
                    image_name_H = os.path.join(self.data_root, row[0] + '_hist.nii.gz')
                    label_name = os.path.join(self.data_root, row[0] + '_Label.nii.gz')
                if row[1] == "training":
                    if self.dataset == "both":
                        train_files.append({"imageT1": image_name_T1,"imageT2": image_name_T2, "label": label_name})
                    elif self.dataset == "both_h":
                        train_files.append({"imageT1": image_name_T1, "imageT2": image_name_T2,"imageH": image_name_H, "label": label_name})
                    else:
                        train_files.append({"image": image_name, "label": label_name})


        # check if all files exist
        # for file_dict in train_files + val_files + test_files:
        #     assert (os.path.isfile(file_dict['image'])), f" {file_dict['image']} is not a file"
        #     assert (os.path.isfile(file_dict['label'])), f" {file_dict['label']} is not a file"
        #
        # test_files
        # return as dictionaries of image/label pairs
        print(train_files)
        return train_files

    def get_transforms(self):
        self.logger.info("Getting transforms...")
        # Setup transforms of data sets
        if self.dataset == "both":
            train_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1",'imageT2', "label"]),
                    AddChanneld(keys=["imageT1",'imageT2', "label"]),
                    Orientationd(keys=["imageT1",'imageT2', "label"], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1",'imageT2'], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1",'imageT2'], threshold=1800, above=False),
                    RandAdjustContrastd(keys=["imageT1",'imageT2'],prob=0.2, gamma=(0.5, 4.5)),
                    NormalizeIntensityd(keys=["imageT1",'imageT2']),
                    SpatialPadd(keys=["imageT1",'imageT2', "label"], spatial_size=self.pad_crop_shape),
                    RandFlipd(keys=["imageT1",'imageT2', "label"], prob=0.5, spatial_axis=0),
                    RandSpatialCropd(
                        keys=["imageT1",'imageT2', "label"], roi_size=self.pad_crop_shape, random_center=True, random_size=False
                    ),
                    RandAffined(keys=["imageT1", 'imageT2', "label"], rotate_range=(0.2, 0.2, 0.2),
                                scale_range=([0.8, 1.2], [0.8, 1.2], [0.8, 1.2]), mode='nearest',prob=0.2),
                    ToTensord(keys=["imageT1",'imageT2', "label"]),
                ]
            )

            train_target_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1", 'imageT2']),
                    AddChanneld(keys=["imageT1", 'imageT2']),
                    Orientationd(keys=["imageT1", 'imageT2'], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1", 'imageT2'], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1", 'imageT2'], threshold=1800, above=False),
                    RandAdjustContrastd(keys=["imageT1", 'imageT2'], prob=0.2, gamma=(0.5, 4.5)),
                    NormalizeIntensityd(keys=["imageT1", 'imageT2']),
                    SpatialPadd(keys=["imageT1", 'imageT2'], spatial_size=self.pad_crop_shape),
                    RandFlipd(keys=["imageT1", 'imageT2'], prob=0.5, spatial_axis=0),
                    RandSpatialCropd(
                        keys=["imageT1", 'imageT2'], roi_size=self.pad_crop_shape, random_center=True,
                        random_size=False
                    ),
                    RandAffined(keys=["imageT1", 'imageT2'], rotate_range=(0.2, 0.2, 0.2),
                                scale_range=([0.8, 1.2], [0.8, 1.2], [0.8, 1.2]), mode='nearest', prob=0.2),
                    ToTensord(keys=["imageT1", 'imageT2']),
                ]
            )

            val_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1",'imageT2', "label"]),
                    AddChanneld(keys=["imageT1",'imageT2', "label"]),
                    Orientationd(keys=["imageT1",'imageT2', "label"], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1",'imageT2'], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1",'imageT2'], threshold=1800, above=False),
                    NormalizeIntensityd(keys=["imageT1",'imageT2']),
                    SpatialPadd(keys=["imageT1",'imageT2', "label"], spatial_size=self.pad_crop_shape),
                    RandSpatialCropd(
                        keys=["imageT1",'imageT2', "label"], roi_size=self.pad_crop_shape, random_center=True, random_size=False,
                    ),
                    ToTensord(keys=["imageT1",'imageT2', "label"]),
                ]
            )

            test_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1",'imageT2']),
                    AddChanneld(keys=["imageT1",'imageT2']),
                    Orientationd(keys=["imageT1",'imageT2'], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1",'imageT2'], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1",'imageT2'], threshold=1800, above=False),
                    NormalizeIntensityd(keys=["imageT1",'imageT2']),
                    ToTensord(keys=["imageT1",'imageT2']),
                ]
            )
        elif self.dataset == "both_h":
            train_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1",'imageT2','imageH', "label"]),
                    AddChanneld(keys=["imageT1",'imageT2','imageH', "label"]),
                    Orientationd(keys=["imageT1",'imageT2','imageH', "label"], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1",'imageT2','imageH'], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1",'imageT2','imageH'], threshold=1800, above=False),
                    RandAdjustContrastd(keys=["imageT1",'imageT2','imageH'],prob=0.2, gamma=(0.5, 4.5)),
                    NormalizeIntensityd(keys=["imageT1",'imageT2','imageH']),
                    SpatialPadd(keys=["imageT1",'imageT2','imageH', "label"], spatial_size=self.pad_crop_shape),
                    RandFlipd(keys=["imageT1",'imageT2','imageH', "label"], prob=0.5, spatial_axis=0),
                    RandSpatialCropd(
                        keys=["imageT1",'imageT2','imageH', "label"], roi_size=self.pad_crop_shape, random_center=True, random_size=False
                    ),
                    RandAffined(keys=["imageT1", 'imageT2','imageH', "label"], rotate_range=(0.2, 0.2, 0.2),
                                scale_range=([0.8, 1.2], [0.8, 1.2], [0.8, 1.2]), mode='nearest',prob=0.2),
                    ToTensord(keys=["imageT1",'imageT2','imageH', "label"]),
                ]
            )

            train_target_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1", 'imageT2','imageH']),
                    AddChanneld(keys=["imageT1", 'imageT2','imageH']),
                    Orientationd(keys=["imageT1", 'imageT2','imageH'], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1", 'imageT2','imageH'], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1", 'imageT2','imageH'], threshold=1800, above=False),
                    RandAdjustContrastd(keys=["imageT1", 'imageT2','imageH'], prob=0.2, gamma=(0.5, 4.5)),
                    NormalizeIntensityd(keys=["imageT1", 'imageT2','imageH']),
                    SpatialPadd(keys=["imageT1", 'imageT2','imageH'], spatial_size=self.pad_crop_shape),
                    RandFlipd(keys=["imageT1", 'imageT2','imageH'], prob=0.5, spatial_axis=0),
                    RandSpatialCropd(
                        keys=["imageT1", 'imageT2','imageH'], roi_size=self.pad_crop_shape, random_center=True,
                        random_size=False
                    ),
                    RandAffined(keys=["imageT1", 'imageT2','imageH'], rotate_range=(0.2, 0.2, 0.2),
                                scale_range=([0.8, 1.2], [0.8, 1.2], [0.8, 1.2]), mode='nearest', prob=0.2),
                    ToTensord(keys=["imageT1", 'imageT2','imageH']),
                ]
            )

            val_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1",'imageT2','imageH', "label"]),
                    AddChanneld(keys=["imageT1",'imageT2','imageH', "label"]),
                    Orientationd(keys=["imageT1",'imageT2','imageH', "label"], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1",'imageT2','imageH'], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1",'imageT2','imageH'], threshold=1800, above=False),
                    NormalizeIntensityd(keys=["imageT1",'imageT2','imageH']),
                    SpatialPadd(keys=["imageT1",'imageT2','imageH', "label"], spatial_size=self.pad_crop_shape),
                    RandSpatialCropd(
                        keys=["imageT1",'imageT2','imageH', "label"], roi_size=self.pad_crop_shape, random_center=True, random_size=False,
                    ),
                    ToTensord(keys=["imageT1",'imageT2','imageH', "label"]),
                ]
            )

            test_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1",'imageT2','imageH']),
                    AddChanneld(keys=["imageT1",'imageT2','imageH']),
                    Orientationd(keys=["imageT1",'imageT2','imageH'], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1",'imageT2','imageH'], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1",'imageT2','imageH'], threshold=1800, above=False),
                    NormalizeIntensityd(keys=["imageT1",'imageT2','imageH']),
                    ToTensord(keys=["imageT1",'imageT2','imageH']),
                ]
            )
        else:
            train_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    NormalizeIntensityd(keys=["image"]),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandZoomd(keys=["image", "label"], prob=0.2, mode='nearest'),
                    RandRotated(keys=["image", "label"], prob=0.2, mode='nearest', range_x=0.2, range_y=0.2,
                                range_z=0.2),
                    (RandCropByLabelClassesd(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=self.pad_crop_shape,
                        ratios=[2, 2, 5],
                        num_classes=3,
                        num_samples=1) if self.args.weighted_crop else Identityd(keys=["image", "label"])),
                    RandSpatialCropd(
                        keys=["image", "label"], roi_size=[self.img_size, self.img_size, 20],
                        max_roi_size=self.pad_crop_shape, random_center=True, random_size=True,
                    ),
                    SpatialPadd(keys=["image", "label"], spatial_size=self.pad_crop_shape),
                    ToTensord(keys=["image", "label"]),
                ]
            )

            if self.args.DA:
                train_target_transforms = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        AddChanneld(keys=["image", "label"]),
                        Orientationd(keys=["image", "label"], axcodes="RAS"),
                        NormalizeIntensityd(keys=["image"]),
                        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                        RandZoomd(keys=["image", "label"], prob=0.2, mode='nearest'),
                        RandRotated(keys=["image", "label"], prob=0.2, mode='nearest', range_x=0.2, range_y=0.2,
                                    range_z=0.2),
                        (RandCropByLabelClassesd(
                            keys=["image", "label"],
                            label_key="label",
                            spatial_size=self.pad_crop_shape,
                            ratios=[2, 2, 5],
                            num_classes=3,
                            num_samples=1) if self.args.weighted_crop else Identityd(keys=["image", "label"])),
                        RandSpatialCropd(
                            keys=["image", "label"], roi_size=[self.img_size, self.img_size, 20],
                            max_roi_size=self.pad_crop_shape, random_center=True, random_size=True,
                        ),
                        SpatialPadd(keys=["image", "label"], spatial_size=self.pad_crop_shape),
                        ToTensord(keys=["image", "label"]),
                    ]
                )
            else:
                train_target_transforms = Compose(
                    [
                        LoadImaged(keys=["image"]),
                        AddChanneld(keys=["image"]),
                        Orientationd(keys=["image"], axcodes="RAS"),
                        NormalizeIntensityd(keys=["image"]),
                        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                        RandZoomd(keys=["image"], prob=0.2, mode='nearest'),
                        RandRotated(keys=["image"], prob=0.2, mode='nearest', range_x=0.2, range_y=0.2,
                                    range_z=0.2),
                        RandSpatialCropd(
                            keys=["image"], roi_size=[self.img_size, self.img_size, 20],
                            max_roi_size=self.pad_crop_shape, random_center=True, random_size=True,
                        ),
                        SpatialPadd(keys=["image"], spatial_size=self.pad_crop_shape),
                        ToTensord(keys=["image"]),
                    ]
                )

            val_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    NormalizeIntensityd(keys=["image"]),
                    ToTensord(keys=["image", "label"]),
                ]
            )

            test_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    NormalizeIntensityd(keys=["image"]),
                    ToTensord(keys=["image"]),
                ]
            )

        return train_transforms, train_target_transforms ,val_transforms, test_transforms

    @staticmethod
    def get_center_of_mass_slice(label):
        # calculate center of mass of label in through plan direction to select a slice that shows the tumour
        num_slices = label.shape[2]
        slice_masses = np.zeros(num_slices)
        for z in range(num_slices):
            slice_masses[z] = label[:, :, z].sum()

        if sum(slice_masses) == 0:  # if there is no label in the cropped image
            slice_weights = np.ones(num_slices) / num_slices  # give all slices equal weight
        else:
            slice_weights = slice_masses / sum(slice_masses)

        center_of_mass = sum(slice_weights * np.arange(num_slices))
        slice_closest_to_center_of_mass = int(center_of_mass.round())
        return slice_closest_to_center_of_mass

    def check_transforms_on_first_validation_image_and_label(self, val_files, val_transforms):
        logger = self.logger
        # check the transforms on the first validation set image
        check_ds = monai.data.Dataset(data=val_files, transform=val_transforms)  # apply transform
        check_loader = DataLoader(check_ds, batch_size=1)
        check_data = monai.utils.misc.first(check_loader)  # gets the first item from an input iterable
        if self.dataset == 'both':
            imageT1,imageT2, label = (check_data["imageT1"][0][0],check_data["imageT2"][0][0], check_data["label"][0][0])
            logger.info("-" * 10)
            logger.info("Check the transforms on the first validation set image and label")
            logger.info(
                "Length of check_data = {}".format(len(check_data))
            )  # this dictionary also contains all the nifti header info
            logger.info("check_data['imageT1'].shape = {}".format(check_data["imageT1"].shape))
            logger.info("Validation image shape = {}".format(imageT1.shape))
            logger.info("check_data['imageT2'].shape = {}".format(check_data["imageT2"].shape))
            logger.info("Validation image shape = {}".format(imageT2.shape))
            logger.info("Validation label shape = {}".format(label.shape))

            slice_idx = self.get_center_of_mass_slice(
                label
            )  # choose slice of selected validation set image volume for the figure

            logger.info("-" * 10)
            logger.info("Plot one slice of the image and the label")
            logger.info("image shape: {}, label shape: {}, slice = {}".format(imageT1.shape, label.shape, slice_idx))
            # plot the slice [:, :, slice]
            plt.figure("check", (12, 6))
            plt.subplot(1, 2, 1)
            plt.title("imageT1")
            plt.imshow(imageT1[:, :, slice_idx], cmap="gray", interpolation="none")
            plt.subplot(1, 2, 2)
            plt.title("label")
            plt.imshow(label[:, :, slice_idx], interpolation="none")
            plt.savefig(os.path.join(self.figures_path, "check_validation_imageT1_and_label.png"))

            logger.info("image shape: {}, label shape: {}, slice = {}".format(imageT2.shape, label.shape, slice_idx))
            # plot the slice [:, :, slice]
            plt.figure("check", (12, 6))
            plt.subplot(1, 2, 1)
            plt.title("imageT2")
            plt.imshow(imageT2[:, :, slice_idx], cmap="gray", interpolation="none")
            plt.subplot(1, 2, 2)
            plt.title("label")
            plt.imshow(label[:, :, slice_idx], interpolation="none")
            plt.savefig(os.path.join(self.figures_path, "check_validation_imageT2_and_label.png"))
        elif self.dataset == 'both_h':
            imageT1,imageT2,imageH, label = (check_data["imageT1"][0][0],check_data["imageT2"][0][0],check_data["imageH"][0][0], check_data["label"][0][0])
            logger.info("-" * 10)
            logger.info("Check the transforms on the first validation set image and label")
            logger.info(
                "Length of check_data = {}".format(len(check_data))
            )  # this dictionary also contains all the nifti header info
            logger.info("check_data['imageT1'].shape = {}".format(check_data["imageT1"].shape))
            logger.info("Validation image shape = {}".format(imageT1.shape))
            logger.info("check_data['imageT2'].shape = {}".format(check_data["imageT2"].shape))
            logger.info("Validation image shape = {}".format(imageT2.shape))
            logger.info("Validation label shape = {}".format(label.shape))

            slice_idx = self.get_center_of_mass_slice(
                label
            )  # choose slice of selected validation set image volume for the figure

            logger.info("-" * 10)
            logger.info("Plot one slice of the image and the label")
            logger.info("image shape: {}, label shape: {}, slice = {}".format(imageT1.shape, label.shape, slice_idx))
            # plot the slice [:, :, slice]
            plt.figure("check", (12, 6))
            plt.subplot(1, 2, 1)
            plt.title("imageT1")
            plt.imshow(imageT1[:, :, slice_idx], cmap="gray", interpolation="none")
            plt.subplot(1, 2, 2)
            plt.title("label")
            plt.imshow(label[:, :, slice_idx], interpolation="none")
            plt.savefig(os.path.join(self.figures_path, "check_validation_imageT1_and_label.png"))

            logger.info("image shape: {}, label shape: {}, slice = {}".format(imageT2.shape, label.shape, slice_idx))
            # plot the slice [:, :, slice]
            plt.figure("check", (12, 6))
            plt.subplot(1, 2, 1)
            plt.title("imageT2")
            plt.imshow(imageT2[:, :, slice_idx], cmap="gray", interpolation="none")
            plt.subplot(1, 2, 2)
            plt.title("label")
            plt.imshow(label[:, :, slice_idx], interpolation="none")
            plt.savefig(os.path.join(self.figures_path, "check_validation_imageT2_and_label.png"))

            logger.info("image shape: {}, label shape: {}, slice = {}".format(imageH.shape, label.shape, slice_idx))
            # plot the slice [:, :, slice]
            plt.figure("check", (12, 6))
            plt.subplot(1, 2, 1)
            plt.title("imageH")
            plt.imshow(imageH[:, :, slice_idx], cmap="gray", interpolation="none")
            plt.subplot(1, 2, 2)
            plt.title("label")
            plt.imshow(label[:, :, slice_idx], interpolation="none")
            plt.savefig(os.path.join(self.figures_path, "check_validation_imageH_and_label.png"))
        else:
            image, label = (check_data["image"][0][0], check_data["label"][0][0])
            logger.info("-" * 10)
            logger.info("Check the transforms on the first validation set image and label")
            logger.info(
                "Length of check_data = {}".format(len(check_data))
            )  # this dictionary also contains all the nifti header info
            logger.info("check_data['image'].shape = {}".format(check_data["image"].shape))
            logger.info("Validation image shape = {}".format(image.shape))
            logger.info("Validation label shape = {}".format(label.shape))

            slice_idx = self.get_center_of_mass_slice(
                label
            )  # choose slice of selected validation set image volume for the figure

            logger.info("-" * 10)
            logger.info("Plot one slice of the image and the label")
            logger.info("image shape: {}, label shape: {}, slice = {}".format(image.shape, label.shape, slice_idx))
            # plot the slice [:, :, slice]
            plt.figure("check", (12, 6))
            plt.subplot(1, 2, 1)
            plt.title("image")
            plt.imshow(image[:, :, slice_idx], cmap="gray", interpolation="none")
            plt.subplot(1, 2, 2)
            plt.title("label")
            plt.imshow(label[:, :, slice_idx], interpolation="none")
            plt.savefig(os.path.join(self.figures_path, "check_validation_image_and_label.png"))

    # Set different seed for workers of DataLoader
    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.transform.set_random_state(worker_info.seed % (2 ** 32))

    def cache_transformed_train_data(self, train_files, train_transforms):
        self.logger.info("Caching training data set...")
        # Define SmartCacheDataset and DataLoader for training and validation
        train_ds = monai.data.CacheDataset(
            data=train_files, transform=train_transforms, cache_rate= self.cache_rate
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=monai.data.list_data_collate,
            worker_init_fn=self.worker_init_fn,
        )
        return train_ds, train_loader


    def cache_transformed_val_data(self, val_files, val_transforms):
        self.logger.info("Caching validation data set...")
        val_ds = monai.data.CacheDataset(
            data=val_files, transform=val_transforms, cache_rate=0
        )
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=self.num_workers)
        return val_loader

    def cache_transformed_test_data(self, test_files, test_transforms):
        self.logger.info("Caching test data set...")
        test_ds = monai.data.CacheDataset(
            data=test_files, transform=test_transforms, cache_rate=0
        )
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=self.num_workers, shuffle=False)
        return test_loader

    def set_and_get_model(self):

        if self.dataset == 'both_h':
            input_dim = 3
        elif self.dataset == 'both':
            input_dim = 2
        else:
            input_dim = 1

        if self.model == "UNet2d5_spvPA":

            model = UNet2d5_spvPA(
                dimensions=3,
                in_channels=(input_dim),
                out_channels=3,
                channels=(24, 32, 48, 64, 80, 96),
                strides=(
                    (2, 2, 1),
                    (2, 2, 1),
                    (2, 2, 2),
                    (2, 2, 2),
                    (2, 2, 2),
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
                norm=Norm.INSTANCE,
                dropout=0.1,
                attention_module=self.attention,
            ).to(self.device)

        elif self.model == 'unet_assp':

            model = unet_assp(in_dim=(input_dim)).to(self.device)

        elif self.model == 'unet_3+':

            model = UNet_3Plus(in_channels=(input_dim), n_classes=3).to(self.device)

        D = FCDiscriminator(num_classes=3)#PatchDiscriminator_3D(input_nc=3).to(self.device)

        # hl.build_graph(model, torch.zeros(2, 1, 128, 128, 32).to(self.device)).save("model")
        if self.args.world_size <= 1:
            return model, D
        else:
            ddp_model = DDP(model, broadcast_buffers=False)
            ddp_D = DDP(D, broadcast_buffers=False)
            return ddp_model, ddp_D

    def set_and_get_loss_function(self):
        self.logger.info("Setting up the loss function...")
        if self.model == "UNet2d5_spvPA":
            loss_function = Dice_spvPA(
                to_onehot_y=True, softmax=True, supervised_attention=self.attention, hardness_weighting=self.hardness
            )
        else:
            loss_function = DiceLoss(to_onehot_y=True, softmax=True, hardness_weight=self.hardness)
        return loss_function

    def set_and_get_optimizer(self, nets):
        self.logger.info("Setting up the optimizer...")
        model, D= nets
        optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_learning_rate,
                                     weight_decay=self.weight_decay)  # torch.optim.SGD(model.parameters(), lr=self.initial_learning_rate, weight_decay=self.weight_decay,momentum=0.9, dampening=0.9)
        optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
        return optimizer, optimizer_D


    def compute_dice_score(self, predicted_probabilities, label):
        y_pred = torch.argmax(predicted_probabilities, dim=1, keepdim=True)  # pick larger value of 2 channels

        y_pred_1 = y_pred*1
        y_pred_1[y_pred_1!=1] = 0
        y_pred_1 = monai.networks.utils.one_hot(y_pred_1, 2)  # make 2 channel one hot tensor

        y_pred_2 = y_pred * 1
        y_pred_2[y_pred_2 != 2] = 0
        y_pred_2[y_pred_2 == 2] = 1
        y_pred_2 = monai.networks.utils.one_hot(y_pred_2, 2)  # make 2 channel one hot tensor


        label_1 = label*1
        label_1[label_1!=1] = 0
        #
        label_2 = label*1
        label_2[label_2 != 2] = 0
        label_2[label_2 == 2] = 1


        dice_score_1 = torch.tensor(
            [
                [
                    1
                    - monai.losses.DiceLoss(
                        include_background=False, to_onehot_y=True, softmax=False, reduction="mean"
                    ).forward(y_pred_1, label_1)
                ]
            ],
            device=self.device,
        )

        dice_score_2 = torch.tensor(
            [
                [
                    1
                    - monai.losses.DiceLoss(
                        include_background=False, to_onehot_y=True, softmax=False, reduction="mean"
                    ).forward(y_pred_2, label_2)
                ]
            ],
            device=self.device,
        )
        return dice_score_1, dice_score_2

    def run_training_algorithm(self, model, loss_function, optimizer, train_loader, val_loader):
        train_ds, train_loader = train_loader
        logger = self.logger
        logger.info("Running the training loop...")
        # TensorBoard tb_Writer will output to ./runs/ directory by default
        tb_writer = SummaryWriter(self.logs_path)

        # add an image grid to tensorboard
        if self.debug:
            images_for_grid = []
            for batch_data in train_loader:
                if self.dataset=='both':
                    imagesT1, imagesT2, labels = batch_data["imageT1"],batch_data["imageT2"], batch_data["label"]
                    images = torch.cat([imagesT1,imagesT2],dim=1)
                elif self.dataset=='both_h':
                    imagesT1, imagesT2, imagesH, labels = batch_data["imageT1"], batch_data["imageT2"], batch_data["imageH"], batch_data["label"]
                    images = torch.cat([imagesT1, imagesT2, imagesH], dim=1)
                else:
                    images, labels = batch_data["image"], batch_data["label"]
                for image, label in zip(images, labels):
                    central_slice_number = self.get_center_of_mass_slice(np.squeeze(label[0, :, :, :]))
                    images_for_grid.append(image[..., central_slice_number])
                    images_for_grid.append(label[..., central_slice_number])
            image_grid = torchvision.utils.make_grid(images_for_grid, normalize=True, scale_each=True)
            tb_writer.add_image("images", image_grid, 0)

        epochs_with_const_lr = self.epochs_with_const_lr
        val_interval = self.val_interval  # validation every val_interval epochs

        # Execute training process
        best_metric_1 = -1  # stores highest mean Dice score obtained during validation
        best_metric_1_epoch = -1  # stores the epoch number during which the highest mean Dice score was obtained
        best_metric_2 = -1  # stores highest mean Dice score obtained during validation
        best_metric_2_epoch = -1
        epoch_loss_values = list()  # stores losses of every epoch
        metric_values = list()  # stores Dice scores of every val_interval epoch
        num_epochs = self.num_epochs
        start = perf_counter()

        for epoch in range(num_epochs):
            logger.info("-" * 10)
            logger.info("Epoch {}/{}".format(epoch + 1, num_epochs))
            if epoch == val_interval:
                stop = perf_counter()
                logger.info(
                    (
                        "Average duration of first {0:.0f} epochs = {1:.2f} s. "
                        + "Expected total training time = {2:.2f} h"
                    ).format(
                        val_interval, (stop - start) / val_interval, (stop - start) * num_epochs / val_interval / 3600
                    )
                )
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1

                if self.dataset=='both':
                    imagesT1, imagesT2, labels = batch_data["imageT1"].to(self.device),batch_data["imageT2"].to(self.device), batch_data["label"].to(self.device)
                    inputs = torch.cat([imagesT1,imagesT2],dim=1)
                elif self.dataset=='both_h':
                    imagesT1, imagesT2, imagesH,labels = batch_data["imageT1"].to(self.device), batch_data["imageT2"].to(
                        self.device), batch_data["imageH"].to(
                        self.device),batch_data["label"].to(self.device)
                    inputs = torch.cat([imagesT1, imagesT2, imagesH], dim=1)
                else:
                    inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
                optimizer.zero_grad()  # reset the optimizer gradient
                t = time.time()
                outputs = model(inputs)  # evaluate the model
                print(time.time()-t)
                t = time.time()
                model(inputs, encode_only=True)
                print(time.time() - t)
                # make_dot(outputs.mean(), params=dict(model.named_parameters())).render("attached", format="png")
                loss = loss_function(outputs, labels)  # returns the mean loss over the batch by default
                loss.backward()  # computes the gradients
                optimizer.step()  # update the model weights
                adjust_learning_rate(lr=self.initial_learning_rate, optimizer=optimizer,batch_size=self.train_batch_size, data_num= len(train_loader), epochs=self.num_epochs)
                epoch_loss += loss.item()
                if epoch == 0:
                    logger.info(
                        "{}/{}, train_loss: {:.4f}".format(step, len(train_loader) // train_loader.batch_size, loss.item())
                    )
            epoch_loss /= step  # calculate mean loss over current epoch
            epoch_loss_values.append(epoch_loss)
            logger.info("epoch {} average loss: {:.4f}".format(epoch + 1, epoch_loss))

            # validation
            if (epoch + 1) % val_interval == 0:
                model.eval()
                if self.model == "UNet2d5_spvPA":
                    model_seg = lambda *args, **kwargs: model(*args, **kwargs)[0]
                with torch.no_grad():  # turns of PyTorch's auto grad for better performance
                    metric_sum_1 = 0.0
                    metric_sum_2 = 0.0
                    metric_count = 0  # counts number of images
                    epoch_loss_val = 0
                    step = 0  # counts number of batches
                    for val_data in val_loader:  # loop over images in validation set
                        step += 1
                        if self.dataset == 'both':
                            val_T1,val_T2, val_labels = val_data["imageT1"].to(self.device), val_data["imageT2"].to(self.device), val_data["label"].to(
                                self.device)
                            val_inputs = torch.cat([val_T1,val_T2],dim=1)
                        elif self.dataset == 'both_h':
                            val_T1, val_T2,val_H, val_labels = val_data["imageT1"].to(self.device), val_data["imageT2"].to(
                                self.device), val_data["imageH"].to(
                                self.device),val_data["label"].to(
                                self.device)
                            val_inputs = torch.cat([val_T1, val_T2,val_H], dim=1)
                        else:
                            val_inputs, val_labels = val_data["image"].to(self.device), val_data["label"].to(
                                self.device)

                        if self.model == "UNet2d5_spvPA":
                            val_outputs = model(val_inputs)[0]
                        else:
                            val_outputs = model(val_inputs)

                        # sliding_outputs = sliding_window_inference(
                        #     inputs=inputs,
                        #     roi_size=self.sliding_window_inferer_roi_size,
                        #     sw_batch_size=1,
                        #     predictor=model_segmentation,
                        #     # mode="gaussian",
                        # )

                        dice_score_1, dice_score_2 = self.compute_dice_score(val_outputs, val_labels)

                        # loss = loss_function(val_outputs, val_labels)

                        metric_count += len(dice_score_1)
                        metric_sum_1 += dice_score_1.sum().item()
                        metric_sum_2 += dice_score_2.sum().item()

                        # epoch_loss_val += loss.item()

                    metric_1 = metric_sum_1 / metric_count  # calculate mean Dice score of current epoch for validation set
                    metric_2 = metric_sum_2 / metric_count
                    metric_values.append(metric_1)
                    metric_values.append(metric_2)
                    epoch_loss_val /= step  # calculate mean loss over current epoch

                    tb_writer.add_scalars("Loss Train/Val", {"train": epoch_loss, "val": epoch_loss_val}, epoch)
                    tb_writer.add_scalar("Dice Score 1 Val", metric_1, epoch)
                    tb_writer.add_scalar("Dice Score 2 Val", metric_2, epoch)
                    if metric_1 > best_metric_1:  # if it's the best Dice score so far, proceed to save
                        best_metric_1 = metric_1
                        best_metric_1_epoch = epoch + 1
                        # save the current best model weights
                        torch.save(model.state_dict(), os.path.join(self.model_path, "best_metric_1_model.pth"))
                        logger.info("saved new best metric1 model")
                    if metric_2 > best_metric_2:  # if it's the best Dice score so far, proceed to save
                        best_metric_2 = metric_2
                        best_metric_2_epoch = epoch + 1
                        # save the current best model weights
                        torch.save(model.state_dict(), os.path.join(self.model_path, "best_metric_2_model.pth"))
                        logger.info("saved new best metric1 model")
                    logger.info(
                        "current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, metric_1, best_metric_1, best_metric_1_epoch
                        )
                    )
                    logger.info(
                        "current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                            epoch + 1, metric_2, best_metric_2, best_metric_2_epoch
                        )
                    )

            torch.save(model.state_dict(), os.path.join(self.model_path, str(epoch)+"_epoch_model.pth"))

            # # learning rate update
            # if (epoch + 1) % epochs_with_const_lr == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group["lr"] = param_group["lr"] / self.lr_divisor
            #         logger.info(
            #             "Dividing learning rate by {}. "
            #             "New learning rate is: lr = {}".format(self.lr_divisor, param_group["lr"])
            #         )

        logger.info("Train completed, best_metric: {:.4f}  at epoch: {}".format(best_metric_1, best_metric_1_epoch))
        logger.info("Train completed, best_metric: {:.4f}  at epoch: {}".format(best_metric_1, best_metric_1_epoch))
        torch.save(model.state_dict(), os.path.join(self.model_path, "last_epoch_model.pth"))
        logger.info(f'Saved model of the last epoch at: {os.path.join(self.model_path, "last_epoch_model.pth")}')
        return epoch_loss_values, metric_values

    def run_training_algorithm_DA(self, nets, loss_function, optimizers, train_loader, target_train_loader, val_loader):
        model, D = nets
        optimizer, optimizer_D = optimizers
        train_ds_s,train_loader = train_loader
        train_ds_t, target_train_loader = target_train_loader
        logger = self.logger
        logger.info("Running the training loop...")
        # TensorBoard tb_Writer will output to ./runs/ directory by default
        tb_writer = SummaryWriter(self.logs_path)

        epochs_with_const_lr = self.epochs_with_const_lr
        val_interval = self.val_interval  # validation every val_interval epochs

        # Execute training process
        best_metric_1 = -1  # stores highest mean Dice score obtained during validation
        best_metric_1_epoch = -1  # stores the epoch number during which the highest mean Dice score was obtained
        best_metric_2 = -1  # stores highest mean Dice score obtained during validation
        best_metric_2_epoch = -1
        epoch_loss_values = list()  # stores losses of every epoch
        metric_values = list()  # stores Dice scores of every val_interval epoch
        num_epochs = self.num_epochs
        start = perf_counter()
        source_label = 0
        target_label = 1

        step = 0

        for epoch in range(num_epochs):
            train_loader_ = torch.utils.data.DataLoader.__iter__(train_loader)
            target_train_loader_ = torch.utils.data.DataLoader.__iter__(target_train_loader)
            logger.info("-" * 10)
            logger.info("Epoch {}/{}".format(epoch + 1, num_epochs))
            if epoch == val_interval:
                stop = perf_counter()
                logger.info(
                    (
                        "Average duration of first {0:.0f} epochs = {1:.2f} s. "
                        + "Expected total training time = {2:.2f} h"
                    ).format(
                        val_interval, (stop - start) / val_interval, (stop - start) * num_epochs / val_interval / 3600
                    )
                )
            model.train()
            D.train()
            epoch_loss = 0
            epoch_loss_target = 0
            epoch_loss_D_s = 0
            epoch_loss_D_t = 0
            sub_step = 0
            for k in range(min(len(train_loader),len(target_train_loader))):
                step += 1
                sub_step += 1

                batch_data = train_loader_.__next__()
                batch_data_target = target_train_loader_.__next__()

                if self.dataset=='both':
                    imagesT1, imagesT2, labels = batch_data["imageT1"].to(self.device),batch_data["imageT2"].to(self.device), batch_data["label"].to(self.device)
                    inputs = torch.cat([imagesT1,imagesT2],dim=1)

                    imagesT1_target, imagesT2_target = batch_data_target["imageT1"].to(self.device), batch_data_target["imageT2"].to(self.device)
                    inputs_target = torch.cat([imagesT1_target, imagesT2_target], dim=1)
                elif self.dataset=='both_h':
                    imagesT1, imagesT2, imagesH,labels = batch_data["imageT1"].to(self.device), batch_data["imageT2"].to(
                        self.device), batch_data["imageH"].to(
                        self.device),batch_data["label"].to(self.device)
                    inputs = torch.cat([imagesT1, imagesT2, imagesH], dim=1)

                    imagesT1_target, imagesT2_target, imagesH_target = batch_data_target["imageT1"].to(self.device), batch_data_target[
                        "imageT2"].to(self.device), batch_data_target[
                        "imageH"].to(self.device)
                    inputs_target = torch.cat([imagesT1_target, imagesT2_target, imagesH_target], dim=1)
                else:
                    inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
                    inputs_target = batch_data_target["image"].to(self.device)

                for param in D.parameters():
                    param.requires_grad = False

                # train with source
                optimizer.zero_grad()
                optimizer_D.zero_grad()


                pred = model(inputs)

                loss = loss_function(pred, labels)

                if self.args.rank == 0:
                    tb_writer.add_scalar('supervised loss',
                                  loss.item(), step)

                loss.backward()
                average_gradients(model)


                if epoch == 0:
                    logger.info(
                        "{}/{}, train_loss: {:.4f}".format(step, len(train_loader) // self.train_batch_size, loss.item())
                    )

                epoch_loss += loss.item()

                # train with target
                if self.model == "UNet2d5_spvPA":
                    pred_target, pred_target_a = model(inputs_target)
                    if self.G_C:
                        dim_index = np.random.choice([2,3,4],2,replace=False)
                        inputs_target_G = inputs_target.transpose(dim_index.max(),dim_index.min())
                        pred_target_G, pred_target_G_a = model(inputs_target_G)
                        pred_target_G = pred_target_G

                        y_p = torch.argmax(pred_target,dim=1,keepdim=True).float()
                        y_p = y_p.transpose(dim_index.max(),dim_index.min())
                        y_G_p = torch.argmax(pred_target_G,dim=1,keepdim=True).float()
                        y_G_p = y_G_p.transpose(dim_index.min(),dim_index.max())
                        loss_G = (loss_function([pred_target,pred_target_a],y_G_p) + loss_function([pred_target_G,pred_target_G_a],y_p))*0.5
                        if self.args.rank == 0:
                            tb_writer.add_scalar('geometry consitency', loss_G, step)
                else:
                    pred_target = model(inputs_target)

                D_out = D(pred_target)

                pred_target_probability = F.softmax(pred_target,dim=1)
                loss_entropy_minization =  (-pred_target_probability * torch.log(pred_target_probability + 1e-5)).sum(1).mean()

                if self.args.rank == 0:
                    tb_writer.add_scalar('EMG loss',
                                  loss_entropy_minization, step)

                loss_adv_target = F.binary_cross_entropy(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(self.device))
                epoch_loss_target += loss_adv_target.item()

                if self.args.rank == 0:
                    tb_writer.add_scalar('loss_adv_target',
                                  loss_adv_target, step)

                loss = 0.001 * loss_adv_target + loss_entropy_minization*(1.0 if self.EMG else 0.0) + (0.1*loss_G if self.G_C else 0)
                loss.backward()
                average_gradients(model)

                # train D

                # bring back requires_grad
                for param in D.parameters():
                    param.requires_grad = True

                # train with source
                if self.model == "UNet2d5_spvPA":
                    pred = pred[0].detach()
                else:
                    pred = pred.detach()

                D_out = D(pred)


                loss_D = F.binary_cross_entropy(D_out,torch.FloatTensor(D_out.data.size()).fill_(source_label).to(self.device))
                epoch_loss_D_s += loss_D.item()
                if self.args.rank == 0:
                    tb_writer.add_scalar('loss Ds',
                                  loss_D, step)


                loss_D.backward()
                average_gradients(D)


                # train with target
                pred_target = pred_target.detach()

                D_out = D(pred_target)

                if self.args.rank == 0:
                    tb_writer.add_scalar('loss Dt',
                                 loss_D, step)

                loss_D = F.binary_cross_entropy(D_out,torch.FloatTensor(D_out.data.size()).fill_(target_label).to(self.device))
                epoch_loss_D_t += loss_D.item()

                loss_D.backward()
                average_gradients(D)

                optimizer.step()
                optimizer_D.step()

            def select_n_random(data, labels, n=1000):
                '''
                Selects n random datapoints and their corresponding labels from a dataset
                '''
                assert len(data) == len(labels)

                perm = torch.randperm(len(data))
                return data[perm][:n], labels[perm][:n]

            features_s = pred.permute(0, 2, 3, 4, 1).reshape(-1, 3)
            features_t = pred_target.permute(0, 2, 3, 4, 1).reshape(-1, 3)
            features = torch.cat([features_s, features_t], dim=0)
            class_labels = torch.cat([torch.ones(features_s.shape[0]), torch.zeros(features_t.shape[0])])
            features, labels = select_n_random(features, class_labels)
            labels = [lab for lab in labels]
            if self.args.rank == 0:
                tb_writer.add_embedding(features,
                                    metadata=labels,global_step=step)

            epoch_loss /= sub_step  # calculate mean loss over current epoch
            epoch_loss_target /= sub_step
            epoch_loss_D_s /= sub_step
            epoch_loss_D_t /= sub_step

            epoch_loss_values.append(epoch_loss)#,epoch_loss_target, epoch_loss_D_s, epoch_loss_D_t
            logger.info("epoch {} average loss: {:.4f} target: {:.4f} Ds: {:.4f} Dt: {:.4f}".format(epoch + 1, epoch_loss, epoch_loss_target, epoch_loss_D_s, epoch_loss_D_t))

            # validation
            if self.args.rank == 0:
                if (epoch) % val_interval == 0:
                    model.eval()
                    with torch.no_grad():  # turns of PyTorch's auto grad for better performance
                        if self.model == "UNet2d5_spvPA":
                            model_segmentation = lambda *args, **kwargs: model(*args, **kwargs)[0]
                        else:
                            model_segmentation = model

                        metric_sum_1 = 0.0
                        metric_sum_2 = 0.0
                        metric_count = 0  # counts number of images
                        epoch_loss_val = 0
                        step_val = 0  # counts number of batches
                        for val_data in val_loader:  # loop over images in validation set
                            step_val += 1
                            if self.dataset == 'both':
                                val_T1, val_T2, val_labels = val_data["imageT1"].to(self.device), val_data[
                                    "imageT2"].to(self.device), val_data["label"].to(
                                    self.device)
                                val_inputs = torch.cat([val_T1, val_T2], dim=1)
                            elif self.dataset == 'both_h':
                                val_T1, val_T2, val_H, val_labels = val_data["imageT1"].to(self.device), val_data[
                                    "imageT2"].to(
                                    self.device), val_data["imageH"].to(
                                    self.device), val_data["label"].to(
                                    self.device)
                                val_inputs = torch.cat([val_T1, val_T2, val_H], dim=1)
                            else:
                                val_inputs, val_labels = val_data["image"].to(self.device), val_data["label"].to(
                                    self.device)

                            if self.model == "UNet2d5_spvPA":
                                val_outputs = sliding_window_inference(
                                    inputs=val_inputs,
                                    roi_size=self.sliding_window_inferer_roi_size,
                                    sw_batch_size=1,
                                    predictor=model_segmentation,
                                    # mode="gaussian",
                                )
                            else:
                                val_outputs = sliding_window_inference(
                                    inputs=val_inputs,
                                    roi_size=self.sliding_window_inferer_roi_size,
                                    sw_batch_size=1,
                                    predictor=model,
                                    # mode="gaussian",
                                )

                            # sliding_outputs = sliding_window_inference(
                            #     inputs=inputs,
                            #     roi_size=self.sliding_window_inferer_roi_size,
                            #     sw_batch_size=1,
                            #     predictor=model_segmentation,
                            #     # mode="gaussian",
                            # )

                            dice_score_1, dice_score_2 = self.compute_dice_score(val_outputs, val_labels)

                            # loss = loss_function(val_outputs, val_labels)

                            metric_count += len(dice_score_1)
                            metric_sum_1 += dice_score_1.sum().item()
                            metric_sum_2 += dice_score_2.sum().item()

                            # epoch_loss_val += loss.item()

                        metric_1 = metric_sum_1 / metric_count  # calculate mean Dice score of current epoch for validation set
                        metric_2 = metric_sum_2 / metric_count
                        metric_values.append(metric_1)
                        metric_values.append(metric_2)
                        epoch_loss_val /= step_val  # calculate mean loss over current epoch

                        tb_writer.add_scalars("Loss Train/Val", {"train": epoch_loss, "val": epoch_loss_val}, epoch)
                        tb_writer.add_scalar("Dice Score 1 Val", metric_1, epoch)
                        tb_writer.add_scalar("Dice Score 2 Val", metric_2, epoch)
                        if metric_1 > best_metric_1:  # if it's the best Dice score so far, proceed to save
                            best_metric_1 = metric_1
                            best_metric_1_epoch = epoch + 1
                            # save the current best model weights
                            torch.save(model.state_dict(), os.path.join(self.model_path, "best_metric_1_model.pth"))
                            logger.info("saved new best metric1 model")
                        if metric_2 > best_metric_2:  # if it's the best Dice score so far, proceed to save
                            best_metric_2 = metric_2
                            best_metric_2_epoch = epoch + 1
                            # save the current best model weights
                            torch.save(model.state_dict(), os.path.join(self.model_path, "best_metric_2_model.pth"))
                            logger.info("saved new best metric1 model")
                        logger.info(
                            "current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                                epoch + 1, metric_1, best_metric_1, best_metric_1_epoch
                            )
                        )
                        logger.info(
                            "current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                                epoch + 1, metric_2, best_metric_2, best_metric_2_epoch
                            )
                        )

                torch.save(model.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_model.pth"))
                torch.save(D.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_D.pth"))

                torch.save(optimizer.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_model_opt.pth"))
                torch.save(optimizer_D.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_D_opt.pth"))

            # # learning rate update
            # if (epoch + 1) % epochs_with_const_lr == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group["lr"] = param_group["lr"] / self.lr_divisor
            #         logger.info(
            #             "Dividing learning rate by {}. "
            #             "New learning rate is: lr = {}".format(self.lr_divisor, param_group["lr"])
            #         )

        if self.args.rank == 0:
            logger.info("Train completed, best_metric: {:.4f}  at epoch: {}".format(best_metric_1, best_metric_1_epoch))
            logger.info("Train completed, best_metric: {:.4f}  at epoch: {}".format(best_metric_1, best_metric_1_epoch))
            torch.save(model.state_dict(), os.path.join(self.model_path, "last_epoch_model.pth"))
            logger.info(f'Saved model of the last epoch at: {os.path.join(self.model_path, "last_epoch_model.pth")}')
            return epoch_loss_values, metric_values

    def plot_loss_curve_and_mean_dice(self, epoch_loss_values, metric_values):
        # Plot the loss and metric
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [self.val_interval * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.savefig(os.path.join(self.figures_path, "epoch_average_loss_and_val_mean_dice.png"))

    def load_trained_state_of_model(self, nets):
        model, _ = nets
        path = self.results_folder_path
        model_path = os.path.join(path, "model")
        dict = torch.load(os.path.join(model_path, "best_metric_2_model.pth"))
        for key in list(dict.keys()):
            new_key = key.replace('module.', '')
            new_key = new_key.replace('sub', 'submodule.')
            dict[new_key] = dict[key]
            del dict[key]
        model.load_state_dict(dict)

        return model

    def run_inference(self, model, data_loader):
        logger = self.logger
        logger.info('Running inference...')

        model.eval()  # activate evaluation mode of model
        dice_scores = np.zeros(len(data_loader))

        if self.model == "UNet2d5_spvPA":
            model_segmentation = lambda *args, **kwargs: model(*args, **kwargs)[0]
        else:
            model_segmentation = model

        with torch.no_grad():  # turns off PyTorch's auto grad for better performance
            for i, data in enumerate(data_loader):
                logger.info("starting image {}".format(i))

                # print(data['image'].shape)
                # outputs = model_segmentation(data["image"].to(self.device))
                # print(outputs.shape)

                if self.dataset == 'both':
                    T1, T2 = data["imageT1"].to(self.device), data["imageT2"].to(
                        self.device)
                    inputs = torch.cat([T1, T2], dim=1)
                elif self.dataset == 'both_h':
                    T1, T2, H = data["imageT1"].to(self.device), data["imageT2"].to(
                        self.device), data["imageH"].to(
                        self.device)
                    inputs = torch.cat([T1, T2, H], dim=1)
                else:
                    inputs = data["image"].to(self.device)

                print(inputs.shape)

                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=self.sliding_window_inferer_roi_size,
                    sw_batch_size=1,
                    predictor=model_segmentation,
                    overlap=0.5,
                    sigma_scale=0.12,
                    mode="gaussian"
                )

                print(outputs.shape)

                # dice_score = self.compute_dice_score(outputs, data["label"].to(self.device))
                # dice_scores[i] = dice_score.item()
                #
                # logger.info(f"dice_score = {dice_score.item()}")

                # export to nifti
                if self.export_inferred_segmentations:
                    logger.info(f"export to nifti...")

                    # print(data[img_dict])

                    nifti_data_matrix = np.squeeze(torch.argmax(outputs, dim=1, keepdim=True))[None, :]
                    if self.dataset =='both':
                        img_dict = 'imageT2_meta_dict'
                    elif self.dataset =='both_h':
                        img_dict = 'imageT2_meta_dict'
                    else:
                        img_dict = 'image_meta_dict'

                    data[img_dict]['filename_or_obj'] = os.path.join('../target_validation',
                                                                                 'crossmoda_' + str(
                                                                                     i + 211) + '_Label.nii.gz')  # data[img_dict]['filename_or_obj'][0]
                    data[img_dict]['affine'] = np.squeeze(data[img_dict]['affine'])
                    data[img_dict]['original_affine'] = np.squeeze(
                        data[img_dict]['original_affine'])

                    print(os.path.join(self.results_folder_path, 'inferred_segmentations_nifti'))
                    saver = NiftiSaver(
                        output_dir=os.path.join(self.results_folder_path, 'inferred_segmentations_nifti'), output_postfix='')
                    saver.save(nifti_data_matrix, meta_data=data[img_dict])
                    print(data[img_dict]['filename_or_obj'], np.unique(nifti_data_matrix.cpu().numpy()))



                    # # plot centre of mass slice of label
                # label = torch.squeeze(data["label"][0, 0, :, :, :])
                # slice_idx = self.get_center_of_mass_slice(
                #     label
                # )  # choose slice of selected validation set image volume for the figure
                # plt.figure("check", (18, 6))
                # plt.clf()
                # plt.subplot(1, 3, 1)
                # plt.title("image " + str(i) + ", slice = " + str(slice_idx))
                # plt.imshow(data["image"][0, 0, :, :, slice_idx], cmap="gray", interpolation="none")
                # plt.subplot(1, 3, 2)
                # plt.title("label " + str(i))
                # plt.imshow(data["label"][0, 0, :, :, slice_idx], interpolation="none")
                # plt.subplot(1, 3, 3)
                # plt.title("output " + str(i) + f", dice = {dice_score.item():.4}")
                # plt.imshow(torch.argmax(outputs, dim=1).detach().cpu()[0, :, :, slice_idx], interpolation="none")
                # plt.savefig(os.path.join(self.figures_path, "best_model_output_val" + str(i) + ".png"))

        # plt.figure("dice score histogram")
        # plt.hist(dice_scores, bins=np.arange(0, 1.01, 0.01))
        # plt.savefig(os.path.join(self.figures_path, "best_model_output_dice_score_histogram.png"))
        #
        # logger.info(f"all_dice_scores = {dice_scores}")
        # logger.info(f"mean_dice_score = {dice_scores.mean()} +- {dice_scores.std()}")