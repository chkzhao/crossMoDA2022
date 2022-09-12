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
import copy
import torch.nn.functional as F
import nibabel as nib
import torch.nn as nn
import time
from .cut_run import CUTModel
from .cut_run_2d import CUTModel_2d
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import SimpleITK as sitk
from .crop_bbox import crop_seg
from scipy.ndimage import zoom
import torch.nn.functional as functional

from params.networks.nets.deeplabv3_3d import DeepLabV3_3D

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

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
    Identityd,
    RandSpatialCropSamplesd
)
from monai.networks.layers import Norm
from monai.data import NiftiSaver
# from torchviz import make_dot
# import hiddenlayer as hl
from .networks.nets.unet2d5_spvPA import UNet2d5_spvPA, UNet2d5_spvPA_T, UNet2d5_spvPA_zoom
from .networks.nets.unet_assp import unet_assp
from .networks.nets.UNet_3Plus import UNet_3Plus
from .networks.nets.transfer_net import PatchSampleF_3D, PatchDiscriminator_3D
from .networks.nets.transfer_net_2d import PatchSampleF, PatchDiscriminator
from .losses.dice_spvPA import Dice_spvPA, DiceLoss
from monai.inferers import sliding_window_inference
from .new_dataset import CacheDataset_v2

monai.config.print_config()


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
        parser.add_argument(
            "--transfer_seg",
            action="store_true",
            help="apply seg loss to transfer",
        )
        parser.add_argument(
            "--load_dict",
            action="store_true",
            help="load_dict",
        )
        parser.add_argument(
            "--weighted_crop",
            action="store_true",
            help="weighted crop label",
        )
        parser.set_defaults(hardness=True)
        parser.add_argument(
            "--results_folder_name", type=str, default="temp" + strftime("%Y%m%d%H%M%S"), help="name of results folder"
        )

        parser.add_argument(
            "--data_root", type=str, default= '', help="name of results folder"
        )
        parser.add_argument(
            "--model", type=str, default='', help="name of model"
        )

        parser.add_argument(
            "--zoom_model", type=str, default='', help="name of zoom_model"
        )

        parser.add_argument('--start_epoch', type=int, default=100, help='world size of distrbuted learning')

        parser.add_argument(
            "--EMG", action='store_true', help='entropy minimization regularization'
        )

        parser.add_argument(
            "--G_C", action='store_true', help='geometry consistency'
        )
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12', help='compute NCE loss on which layers')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch', default=False,
                            help='(used for single image translation) If True')
        parser.add_argument('--direction', type=str, default='AtoB', choices=['AtoB', 'BtoA'],
                            help='transfer direction')

        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--patchD_size', type=int, default=16)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256*4, help='number of patches per layer')
        parser.set_defaults(nce_idt=True, lambda_NCE=4.0)

        parser.add_argument('--world_size', type=int, default=4, help='world size of distrbuted learning')
        parser.add_argument('--rank', type=int, default=0, help='rank of distrbuted learning')
        parser.add_argument('--master_port', type=str, default='69280', help='rank of distrbuted learning')

        parser.add_argument("--warm_seg", type=int, default=202, help="learning rate at first epoch")
        parser.add_argument("--warm_transfer", type=int, default=202, help="learning rate at first epoch")
        parser.add_argument(
            "--no_seg", action='store_true', help='no segmentation'
        )

        parser.add_argument(
            "--zoom_in", action='store_true', help='no segmentation'
        )
        parser.add_argument(
            "--seg_only", action='store_true', help='only segmentation'
        )
        parser.add_argument(
            "--sync_bn", action='store_true', help='ddp sync bn'
        )

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
        self.rank = args.rank
        self.world_size = args.world_size
        self.master_port = args.master_port
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
        self.num_workers = 32
        self.torch_device_arg = "cuda:0"
        self.train_batch_size = args.train_batch_size
        self.initial_learning_rate = args.initial_learning_rate
        self.epochs_with_const_lr = 100
        if self.debug:
            self.epochs_with_const_lr = 3
        self.lr_divisor = 2.0
        self.weight_decay = 1e-7
        self.num_epochs = 150
        # self.num_epochs = int(1.2*self.num_epochs/self.world_size)
        if self.debug:
            self.num_epochs = 10
        self.val_interval = 2  # determines how frequently validation is performed during training
        self.model = args.model #'unet_assp' #"UNet"#"UNet2d5_spvPA"#
        self.sliding_window_inferer_roi_size = [self.img_size, self.img_size, slice_num]
        if self.debug:
            self.sliding_window_inferer_roi_size = [128, 128, slice_num]
        self.attention = args.attention
        self.hardness = args.hardness
        self.export_inferred_segmentations = True

        # paths
        # self.results_folder_path = os.path.join("../results",self.dataset, self.model, args.results_folder_name) #self.data_root
        # if self.debug:
        #     self.results_folder_path = os.path.join(self.data_root, "results", "debug")
        # self.logs_path = os.path.join(self.results_folder_path, "logs")
        # self.model_path = os.path.join(self.results_folder_path, "model")
        # self.figures_path = os.path.join(self.results_folder_path, "figures")
        self.results_folder_path = os.path.join("../results", self.dataset, self.model + self.args.zoom_model,
                                             str(self.args.warm_seg) + '_' + str(self.args.warm_transfer) + '_' + str(
                                                 self.args.intensity) + 'Transfer_' + (
                                                 'Seg_' if self.args.transfer_seg else '') +
                                             ('w_crop_' if self.args.weighted_crop else '') + (
                                                 'dist_' if self.args.world_size >= 1 else '') +('sync_bn_' if self.args.sync_bn else '') +
                                            ('no_seg_' if self.args.no_seg else '') + ('seg_only_' if self.args.seg_only else '') +
                                                ('zoom_in_' if self.args.zoom_in else '') +
                                                (((str(self.args.num_patches) + '_') +
                                                (self.args.direction + '_') + (str(self.args.patchD_size) + '_')) if not self.args.seg_only else '') +
                                              ('noload_' ) +'DA_INS_all_' + self.results_folder_name)

        self.logs_path = os.path.join(self.results_folder_path, "logs")
        self.model_path = os.path.join(self.results_folder_path, "model")
        self.figures_path = os.path.join(self.results_folder_path, "figures")

        #
        self.device = torch.device(self.torch_device_arg)

        self.upsample_img = torch.nn.Upsample(scale_factor=2, mode='trilinear',align_corners=True)
        self.upsample_label = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.crop_cc = RandSpatialCropSamplesd(
                        keys=["image", 'label'], roi_size=[64, 64, 16],num_samples = 2,
                        max_roi_size=self.pad_crop_shape, random_size=True,random_center=True
                    )

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
                elif self.dataset == 'both_sep':
                    image_name_T1 = os.path.join(self.data_root, row[0]+ '_ceT1.nii.gz')
                    image_name_T2 = os.path.join(self.data_root, row[0] + '_hrT2.nii.gz')
                    label_name = os.path.join(self.data_root, row[0] + '_Label.nii.gz')
                else:
                    image_name = os.path.join(self.data_root, row[0] + '_ceT1.nii.gz')
                    label_name = os.path.join(self.data_root, row[0] + '_Label.nii.gz')
                if row[1] == "training":
                    if self.dataset == 'both_sep':
                        train_files.append({"image": image_name_T1, "label": label_name})
                        train_files.append({"image": image_name_T2, "label": label_name})
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
        if self.dataset == 'both':
            train_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1", "imageT2", "label"]),
                    AddChanneld(keys=["imageT1", "imageT2", "label"]),
                    Orientationd(keys=["imageT1", "imageT2", "label"], axcodes="RAS"),
                    RandFlipd(keys=["imageT1", "imageT2", "label"], prob=0.5, spatial_axis=0),
                    RandZoomd(keys=["imageT1", "imageT2", "label"], prob=0.2, mode='nearest'),
                    RandRotated(keys=["imageT1", "imageT2", "label"], prob=0.2, mode='nearest', range_x=0.2,
                                range_y=0.2,
                                range_z=0.2),
                    (RandCropByLabelClassesd(
                        keys=["imageT1", "imageT2", "label"],
                        label_key="label",
                        spatial_size=self.pad_crop_shape,
                        ratios=[3, 3, 6],
                        num_classes=3,
                        num_samples=1) if self.args.weighted_crop else Identityd(keys=["imageT1", "imageT2", "label"])),
                    RandSpatialCropd(
                        keys=["imageT1", "imageT2", "label"], roi_size=[self.img_size, self.img_size, 20],
                        max_roi_size=self.pad_crop_shape, random_size=True, random_center=True,
                    ),
                    SpatialPadd(keys=["imageT1", "imageT2", "label"], spatial_size=self.pad_crop_shape),
                    ThresholdIntensityd(keys=["imageT1", "imageT2"], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1", "imageT2"], threshold=self.args.intensity, above=False),
                    NormalizeIntensityd(keys=["imageT1", "imageT2"], subtrahend=self.args.intensity / 2,
                                        divisor=self.args.intensity / 2),
                    (NormalizeIntensityd(keys=["imageT1",'imageT2']) if self.args.seg_only else Identityd(keys=["imageT1", "imageT2"]) ),
                    ToTensord(keys=["imageT1", "imageT2", "label"]),
                ]
            )

            train_target_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1", "imageT2"]),
                    AddChanneld(keys=["imageT1", "imageT2"]),
                    Orientationd(keys=["imageT1", "imageT2"], axcodes="RAS"),
                    RandFlipd(keys=["imageT1", "imageT2"], prob=0.5, spatial_axis=0),
                    RandZoomd(keys=["imageT1", "imageT2"], prob=0.2, mode='nearest'),
                    RandRotated(keys=["imageT1", "imageT2"], prob=0.2, mode='nearest', range_x=0.2, range_y=0.2,
                                range_z=0.2),
                    RandSpatialCropd(
                        keys=["imageT1", "imageT2"], roi_size=[self.img_size, self.img_size, 20],
                        max_roi_size=self.pad_crop_shape, random_size=True, random_center=True,
                    ),
                    SpatialPadd(keys=["imageT1", "imageT2"], spatial_size=self.pad_crop_shape),
                    ThresholdIntensityd(keys=["imageT1", "imageT2"], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1", "imageT2"], threshold=self.args.intensity, above=False),
                    NormalizeIntensityd(keys=["imageT1", "imageT2"], subtrahend=self.args.intensity / 2,
                                        divisor=self.args.intensity / 2),
                    (NormalizeIntensityd(keys=["imageT1", 'imageT2']) if self.args.seg_only else Identityd(
                        keys=["imageT1", "imageT2"])),
                    ToTensord(keys=["imageT1", "imageT2"]),
                ]
            )

            val_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1", "imageT2", "label"]),
                    AddChanneld(keys=["imageT1", "imageT2", "label"]),
                    Orientationd(keys=["imageT1", "imageT2", "label"], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1", "imageT2"], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1", "imageT2"], threshold=self.args.intensity, above=False),
                    NormalizeIntensityd(keys=["imageT1", "imageT2"], subtrahend=self.args.intensity / 2,
                                        divisor=self.args.intensity / 2),
                    (NormalizeIntensityd(keys=["imageT1", 'imageT2']) if self.args.seg_only else Identityd(
                        keys=["imageT1", "imageT2"])),
                    ToTensord(keys=["imageT1", "imageT2", "label"]),
                ]
            )

            test_transforms = Compose(
                [
                    LoadImaged(keys=["imageT1", "imageT2"]),
                    AddChanneld(keys=["imageT1", "imageT2"]),
                    Orientationd(keys=["imageT1", "imageT2"], axcodes="RAS"),
                    ThresholdIntensityd(keys=["imageT1", "imageT2"], threshold=0, above=True),
                    ThresholdIntensityd(keys=["imageT1", "imageT2"], threshold=self.args.intensity, above=False),
                    NormalizeIntensityd(keys=["imageT1", "imageT2"], subtrahend=self.args.intensity / 2,
                                        divisor=self.args.intensity / 2),
                    (NormalizeIntensityd(keys=["imageT1", 'imageT2']) if self.args.seg_only else Identityd(
                        keys=["imageT1", "imageT2"])),
                    ToTensord(keys=["imageT1", "imageT2"]),
                ]
            )
        else:
            train_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 4.5)),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandZoomd(keys=["image", "label"],prob=0.2,mode='nearest'),
                    RandRotated(keys=["image", "label"],prob=0.2,mode='nearest',range_x=0.2, range_y=0.2,range_z=0.2),
                    (RandCropByLabelClassesd(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=self.pad_crop_shape,
                        ratios=[3, 3, 5],
                        num_classes=3,
                        num_samples=1) if self.args.weighted_crop else Identityd(keys=["image", "label"])),
                    RandSpatialCropd(
                        keys=["image", "label"], roi_size=[self.img_size, self.img_size, 20] ,max_roi_size= self.pad_crop_shape, random_size=True,random_center=True,
                    ),
                    SpatialPadd(keys=["image", "label"], spatial_size=self.pad_crop_shape),
                    ThresholdIntensityd(keys=["image"], threshold=0, above=True),
                    ThresholdIntensityd(keys=["image"], threshold=self.args.intensity, above=False),
                    NormalizeIntensityd(keys=["image"], subtrahend=self.args.intensity / 2,
                                        divisor=self.args.intensity / 2),
                    (NormalizeIntensityd(keys=["image"]) if self.args.seg_only else Identityd(
                        keys=["image"])),
                    ToTensord(keys=["image", "label"]),
                ]
            )

            train_target_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 4.5)),
                    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                    RandZoomd(keys=["image"], prob=0.2, mode='nearest'),
                    RandRotated(keys=["image"], prob=0.2, mode='nearest', range_x=0.2, range_y=0.2,
                                range_z=0.2),
                    RandSpatialCropd(
                        keys=["image"], roi_size=[self.img_size, self.img_size, 20],
                        max_roi_size=self.pad_crop_shape, random_size=True,random_center=True,
                    ),
                    SpatialPadd(keys=["image"], spatial_size=self.pad_crop_shape),
                    ThresholdIntensityd(keys=["image"], threshold=0, above=True),
                    ThresholdIntensityd(keys=["image"], threshold=self.args.intensity, above=False),
                    NormalizeIntensityd(keys=["image"], subtrahend=self.args.intensity / 2,
                                        divisor=self.args.intensity / 2),
                    (NormalizeIntensityd(keys=["image"]) if self.args.seg_only else Identityd(
                        keys=["image"])),
                    ToTensord(keys=["image"]),
                ]
            )

            val_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"]),
                    AddChanneld(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    ThresholdIntensityd(keys=["image"], threshold=0, above=True),
                    ThresholdIntensityd(keys=["image"], threshold=self.args.intensity, above=False),
                    NormalizeIntensityd(keys=["image"], subtrahend=self.args.intensity / 2,
                                        divisor=self.args.intensity / 2),
                    (NormalizeIntensityd(keys=["image"]) if self.args.seg_only else Identityd(
                        keys=["image"])),
                    ToTensord(keys=["image", "label"]),
                ]
            )

            test_transforms = Compose(
                [
                    LoadImaged(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    ThresholdIntensityd(keys=["image"], threshold=0, above=True),
                    ThresholdIntensityd(keys=["image"], threshold=self.args.intensity, above=False),
                    NormalizeIntensityd(keys=["image"], subtrahend=self.args.intensity/2, divisor=self.args.intensity/2),
                    (NormalizeIntensityd(keys=["image"]) if self.args.seg_only else Identityd(
                        keys=["image"])),
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
    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset.transform.set_random_state(worker_info.seed + self.rank)

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

        input_dim = 1

        if self.args.zoom_in:
            input_dim = input_dim * 4

        if self.model == "UNet2d5_spvPA":
            model = UNet2d5_spvPA(
                zoom_in=self.args.zoom_in,
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
                attention_module=self.attention,
            ).to(self.device)

            model_val = UNet2d5_spvPA(
                zoom_in=self.args.zoom_in,
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
                attention_module=self.attention,
            ).to(self.device)

            G = UNet2d5_spvPA_T(
                dimensions=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 48, 64, 80, 96),
                strides=(
                    (2, 2, 1),
                    (2, 2, 1),
                    (2, 2, 2),
                    (2, 2, 2),
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
                attention_module=self.attention,
            ).to(self.device)

        elif self.model == 'unet_assp':

            model = unet_assp(in_dim=(input_dim)).to(self.device)

        elif self.model == 'unet_3+':

            model = UNet_3Plus(in_channels=(input_dim), n_classes=3).to(self.device)
            G = UNet_3Plus(in_channels=(input_dim), n_classes=input_dim).to(self.device)

        D = PatchDiscriminator_3D(input_nc = 1, size=self.args.patchD_size).to(self.device)
        F = PatchSampleF_3D().to(self.device)
        if self.args.zoom_model == "UNet2d5_spvPA":
            zoom_model = UNet2d5_spvPA_zoom(
                zoom_in=self.args.zoom_in,
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
                attention_module=self.attention,
            ).to(self.device)

            zoom_model_val = UNet2d5_spvPA_zoom(
                zoom_in=self.args.zoom_in,
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
                attention_module=self.attention,
            ).to(self.device)
        else:
            resnet = 'resnet18_os16'
            last_activation = 'None'
            zoom_model = DeepLabV3_3D(num_classes=2, input_channels=1, resnet=resnet,
                                      last_activation=last_activation).to(self.device)
            zoom_model_val = DeepLabV3_3D(num_classes=2, input_channels=1, resnet=resnet,
                                          last_activation=last_activation).to(
                self.device)

        process_group = torch.distributed.new_group()

        if self.args.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model,process_group)
            G = nn.SyncBatchNorm.convert_sync_batchnorm(G,process_group)
            D = nn.SyncBatchNorm.convert_sync_batchnorm(D,process_group)
            zoom_model = nn.SyncBatchNorm.convert_sync_batchnorm(zoom_model,process_group)

        ddp_model = DDP(model, broadcast_buffers=True)
        ddp_G = DDP(G, broadcast_buffers=True)
        ddp_D = DDP(D, broadcast_buffers=True)
        ddp_zoom_model = DDP(zoom_model, broadcast_buffers=True)
        if self.args.load_dict:
            path = self.results_folder_path
            model_path = os.path.join(path, "model")
            if not self.args.no_seg:
                dict = torch.load(os.path.join(model_path, str(self.args.start_epoch) + "_epoch_model.pth"))
                ddp_model.load_state_dict(dict)
                dict_zoom = torch.load(os.path.join(model_path, str(self.args.start_epoch) + "_epoch_zoom_model.pth"))
                ddp_zoom_model.load_state_dict(dict_zoom)
            if not self.args.seg_only:
                dict_D = torch.load(os.path.join(model_path, str(self.args.start_epoch) + '_epoch_D.pth'))
                dict_G = torch.load(os.path.join(model_path, str(self.args.start_epoch) + '_epoch_G.pth'))
                ddp_D.load_state_dict(dict_D)
                ddp_G.load_state_dict(dict_G)
        return ddp_model,ddp_zoom_model, ddp_G, ddp_D, F, model_val, zoom_model_val


    def set_and_get_loss_function(self):
        self.logger.info("Setting up the loss function...")
        if self.model == "UNet2d5_spvPA":
            loss_function = Dice_spvPA(
                to_onehot_y=True, softmax=True, supervised_attention=self.attention, hardness_weighting=self.hardness
            )
            loss_function_zoom = DiceLoss(to_onehot_y=True, softmax=True, hardness_weight=self.hardness)
        return loss_function, loss_function_zoom

    def set_and_get_optimizer(self, nets):
        self.logger.info("Setting up the optimizer...")
        model, zoom_model, G, D, F, _, _ = nets
        optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_learning_rate, weight_decay=self.weight_decay)
            #torch.optim.SGD(model.parameters(), lr=self.initial_learning_rate, weight_decay=self.weight_decay,momentum=0.9, dampening=0.9)#
        optimizer_zoom = torch.optim.Adam(zoom_model.parameters(), lr=self.initial_learning_rate, weight_decay=self.weight_decay)
        optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

        if self.args.load_dict:
            path = self.results_folder_path
            model_path = os.path.join(path, "model")
            if not self.args.no_seg:
                dict = torch.load(os.path.join(model_path, str(self.args.start_epoch) + '_epoch_model_opt.pth'))
                optimizer.load_state_dict(dict)
                dict_zoom = torch.load(os.path.join(model_path, str(self.args.start_epoch) + '_epoch_model_zoom_opt.pth'))
                optimizer_zoom.load_state_dict(dict_zoom)
            if not self.args.seg_only:
                dict_D = torch.load(os.path.join(model_path, str(self.args.start_epoch) + '_epoch_D_opt.pth'))
                dict_G = torch.load(os.path.join(model_path, str(self.args.start_epoch) + '_epoch_G_opt.pth'))

                optimizer_D.load_state_dict(dict_D)
                optimizer_G.load_state_dict(dict_G)

        return optimizer,optimizer_zoom, optimizer_G, optimizer_D


    def compute_dice_score(self, y_pred, label):
          # pick larger value of 2 channels

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

    def run_training_algorithm(self, nets, loss_functions, train_loader,
                                  target_train_loader, val_loader):

        model, zoom_model, G, D, F, model_val, zoom_model_val = nets
        optimizer, optimizer_zoom, optimizer_G, optimizer_D = self.set_and_get_optimizer(nets)

        loss_function, loss_function_zoom = loss_functions

        if self.args.transfer_seg:
            transfer_container = CUTModel(self.args, (G, D, F, model, loss_functions[0]), optimizer_G, optimizer_D)
        else:
            transfer_container = CUTModel(self.args, (G, D, F), optimizer_G, optimizer_D)

        train_ds, train_loader = train_loader
        target_train_ds, target_train_loader = target_train_loader
        if self.rank == 0:
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
        iter = 0
        scaler = torch.cuda.amp.GradScaler()
        autocast = torch.cuda.amp.autocast()

        model.train()
        if not self.args.seg_only:
            G.train()
            D.train()
            F.train()

        for epoch in range(self.args.start_epoch, num_epochs):
            epoch_loss = 0
            # if self.rank == 0:
            #     logger.info("-" * 10)
            #     logger.info("Epoch {}/{}".format(epoch + 1, num_epochs))
            #     if epoch == val_interval:
            #         stop = perf_counter()
            #         logger.info(
            #             (
            #                     "Average duration of first {0:.0f} epochs = {1:.2f} s. "
            #                     + "Expected total training time = {2:.2f} h"
            #             ).format(
            #                 val_interval, (stop - start) / val_interval,
            #                               (stop - start) * num_epochs / val_interval / 3600
            #             )
            #         )
            #     # validation
            #     if ((epoch) % val_interval == 0) and not self.args.no_seg:
            #         model_val.load_state_dict(model.module.state_dict())
            #         model_val.eval()
            #
            #         zoom_model_val.load_state_dict(zoom_model.module.state_dict())
            #         zoom_model_val.eval()
            #
            #         with torch.no_grad():  # turns of PyTorch's auto grad for better performance
            #             if self.model == "UNet2d5_spvPA":
            #                 model_segmentation = lambda *args, **kwargs: model_val (*args, **kwargs)[0]
            #             else:
            #                 model_segmentation = model_val
            #
            #             if self.args.zoom_model == "UNet2d5_spvPA":
            #                 model_segmentation_val = lambda *args, **kwargs: zoom_model_val(*args, **kwargs)[0]
            #             else:
            #                 model_segmentation_val = zoom_model_val
            #
            #
            #             metric_sum_1 = 0.0
            #             metric_sum_2 = 0.0
            #             metric_count = 0  # counts number of images
            #             epoch_loss_val = 0
            #             step = 0  # counts number of batches
            #             for val_data in val_loader:  # loop over images in validation set
            #                 step += 1
            #                 if self.dataset == 'both':
            #                     val_T1, val_T2, val_labels = val_data["imageT1"].to(self.device), val_data[
            #                         "imageT2"].to(self.device), val_data["label"].to(
            #                         self.device)
            #                     val_inputs = val_T1
            #                 elif self.dataset == 'both_h':
            #                     val_T1, val_T2, val_H, val_labels = val_data["imageT1"].to(self.device), val_data[
            #                         "imageT2"].to(
            #                         self.device), val_data["imageH"].to(
            #                         self.device), val_data["label"].to(
            #                         self.device)
            #                     val_inputs = torch.cat([val_T1, val_T2, val_H], dim=1)
            #                 else:
            #                     val_inputs, val_labels = val_data["image"].to(self.device), val_data["label"].to(
            #                         self.device)
            #
            #                 if self.model == "UNet2d5_spvPA":
            #                     val_outputs = sliding_window_inference(
            #                         inputs=val_inputs,
            #                         roi_size=self.sliding_window_inferer_roi_size,
            #                         sw_batch_size=1,
            #                         predictor=model_segmentation,
            #                         mode="gaussian",
            #                     )
            #
            #                 else:
            #                     val_outputs = sliding_window_inference(
            #                         inputs=val_inputs,
            #                         roi_size=self.sliding_window_inferer_roi_size,
            #                         sw_batch_size=1,
            #                         predictor=model_segmentation,
            #                         mode="gaussian",
            #                     )
            #
            #                 y_pred_val = torch.argmax(val_outputs, dim=1, keepdim=True)
            #
            #                 seg_map = y_pred_val.squeeze()*1
            #                 seg_map[seg_map!=2] = 0
            #                 seg_map[seg_map==2] = 1
            #
            #                 bbox_lists = crop_seg(seg_map.cpu().numpy(), coc_size=[64,64,32], center=True)
            #
            #                 for bbox in bbox_lists:
            #                     zoom_val_inputs = val_inputs[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            #                     zoom_y_pred_val = y_pred_val[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
            #                     zoom_val_outputs = sliding_window_inference(
            #                         inputs=zoom_val_inputs,
            #                         roi_size=[32,32,16],
            #                         sw_batch_size=1,
            #                         predictor=model_segmentation_val,
            #                         mode="gaussian",
            #                     )
            #
            #                     zoom_y_pred_val_sub = torch.argmax(zoom_val_outputs, dim=1, keepdim=True)
            #
            #                     zoom_y_pred_val[zoom_y_pred_val_sub==1] = 2
            #                     y_pred_val[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = zoom_y_pred_val
            #
            #
            #
            #
            #                 dice_score_1, dice_score_2 = self.compute_dice_score(y_pred_val, val_labels)
            #
            #
            #                 # loss = loss_function(val_outputs, val_labels)
            #
            #                 metric_count += len(dice_score_1)
            #                 metric_sum_1 += dice_score_1.sum().item()
            #                 metric_sum_2 += dice_score_2.sum().item()
            #
            #                 # epoch_loss_val += loss.item()
            #
            #             metric_1 = metric_sum_1 / metric_count  # calculate mean Dice score of current epoch for validation set
            #             metric_2 = metric_sum_2 / metric_count
            #             metric_values.append(metric_1)
            #             metric_values.append(metric_2)
            #             epoch_loss_val /= step  # calculate mean loss over current epoch
            #
            #             tb_writer.add_scalars("Loss Train/Val", {"train": epoch_loss, "val": epoch_loss_val}, epoch)
            #             tb_writer.add_scalar("Dice Score 1 Val", metric_1, epoch)
            #             tb_writer.add_scalar("Dice Score 2 Val", metric_2, epoch)
            #
            #             if metric_1 > best_metric_1:  # if it's the best Dice score so far, proceed to save
            #                 best_metric_1 = metric_1
            #                 best_metric_1_epoch = epoch + 1
            #                 # save the current best model weights
            #                 torch.save(model.state_dict(), os.path.join(self.model_path, "best_metric_1_model.pth"))
            #                 logger.info("saved new best metric1 model")
            #             if metric_2 > best_metric_2:  # if it's the best Dice score so far, proceed to save
            #                 best_metric_2 = metric_2
            #                 best_metric_2_epoch = epoch + 1
            #                 # save the current best model weights
            #                 torch.save(model.state_dict(), os.path.join(self.model_path, "best_metric_2_model.pth"))
            #                 logger.info("saved new best metric1 model")
            #             logger.info(
            #                 "current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
            #                     epoch + 1, metric_1, best_metric_1, best_metric_1_epoch
            #                 )
            #             )
            #             logger.info(
            #                 "current epoch {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
            #                     epoch + 1, metric_2, best_metric_2, best_metric_2_epoch
            #                 )
            #             )
            step = 0
            if self.args.seg_only:
                zip_loader = train_loader
            else:
                zip_loader = zip(train_loader,target_train_loader)
            for batch in zip_loader:

                if self.args.seg_only:
                    batch_data = batch
                else:
                    batch_data, batch_data_target = batch
                step += 1
                iter += 1

                if self.dataset == 'both':
                    imagesT1, imagesT2, labels = batch_data["imageT1"].to(self.device), batch_data["imageT2"].to(
                        self.device), batch_data["label"].to(self.device)
                    inputs = torch.cat([imagesT1, imagesT2], dim=0)

                    labels = torch.cat([labels,labels],dim=0)

                    if not self.args.seg_only:
                        imagesT1_target, imagesT2_target = batch_data_target["imageT1"].to(self.device), \
                                                           batch_data_target[
                                                               "imageT2"].to(self.device)
                        inputs_target = torch.cat([imagesT1_target, imagesT2_target], dim=0)
                elif self.dataset == 'both_h':
                    imagesT1, imagesT2, imagesH, labels = batch_data["imageT1"].to(self.device), batch_data[
                        "imageT2"].to(
                        self.device), batch_data["imageH"].to(
                        self.device), batch_data["label"].to(self.device)
                    inputs = torch.cat([imagesT1, imagesT2, imagesH], dim=1)

                    imagesT1_target, imagesT2_target, imagesH_target = batch_data_target["imageT1"].to(self.device), \
                                                                       batch_data_target[
                                                                           "imageT2"].to(self.device), \
                                                                       batch_data_target[
                                                                           "imageH"].to(self.device)
                    inputs_target = torch.cat([imagesT1_target, imagesT2_target, imagesH_target], dim=1)
                else:
                    inputs, labels = batch_data["image"].to(self.device), batch_data["label"].to(self.device)
                    if not self.args.seg_only:
                        inputs_target = batch_data_target["image"].to(self.device)

                if iter == 1 and not self.args.seg_only:
                    transfer_container.data_dependent_initialize((inputs,inputs_target,labels,epoch))
                    print('finish initializing F')
                    if self.args.load_dict:
                        path = self.results_folder_path
                        model_path = os.path.join(path, "model")
                        dict_F = torch.load(os.path.join(model_path, str(self.args.start_epoch) + '_epoch_F.pth'))
                        dict_F_opt = torch.load(os.path.join(model_path, str(self.args.start_epoch) + '_epoch_F_opt.pth'))
                        F.load_state_dict(dict_F)
                        transfer_container.optimizer_F.load_state_dict(dict_F_opt)
                    F.train()

                if self.args.seg_only:
                    pass
                else:
                    if self.args.direction == 'AtoB':
                        transfer_container.set_input((inputs, inputs_target, labels, epoch))
                    elif self.args.direction == 'BtoA':
                        transfer_container.set_input((inputs_target, inputs, labels, epoch))

                    transfer_container.optimize_parameters()

                    if self.rank == 0:
                        tb_writer.add_scalar('loss_D_real', transfer_container.loss_D_real.item(), iter)
                        tb_writer.add_scalar('loss_D_fake', transfer_container.loss_D_fake.item(), iter)
                        tb_writer.add_scalar('loss_G_GAN', transfer_container.loss_G_GAN.item(), iter)
                        tb_writer.add_scalar('loss_NCE', transfer_container.loss_NCE.item(), iter)
                        tb_writer.add_scalar('loss_NCE_Y', transfer_container.loss_NCE_Y.item(), iter)
                        tb_writer.add_scalar('loss_fake_seg', transfer_container.loss_fake_seg.item(), iter)
                if self.args.no_seg:
                    pass
                else:
                    optimizer.zero_grad()  # reset the optimizer gradient
                    optimizer_zoom.zero_grad()
                    if epoch >= self.args.warm_seg and (not self.args.seg_only):
                        inputs = torch.cat([inputs, transfer_container.fake_B.detach()], dim=0)  # evaluate the model
                        labels = torch.cat([labels, labels], dim=0)

                    with autocast:
                        outputs = model(inputs)
                    loss = loss_function(outputs, labels)  # returns the mean loss over the batch by default
                    scaler.scale(loss).backward()
                    average_gradients(model)

                    if self.rank == 0:
                        tb_writer.add_scalar('loss_seg', loss.item(), iter)
                    scaler.step(optimizer)
                    scaler.update()

                    zoom_inputs = []
                    zoom_labels = []

                    labels[labels!=2] = 0
                    labels[labels==2] = 1

                    for i in range(labels.shape[0]):
                        bbox_lists = crop_seg(labels[i].squeeze().cpu().numpy())
                        for bbox in bbox_lists:
                            zoom_inputs.append(inputs[i, :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]].unsqueeze(dim=0))
                            zoom_labels.append(labels[i, :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]].unsqueeze(dim=0))
                    zoom_inputs = torch.cat(zoom_inputs, dim=0)
                    zoom_labels = torch.cat(zoom_labels, dim=0)

                    with autocast:
                        outputs_zoom = zoom_model(zoom_inputs)

                    if self.args.zoom_model == "UNet2d5_spvPA":
                        loss_zoom = loss_function(outputs_zoom, zoom_labels)
                    else:
                        loss_zoom = loss_function_zoom(outputs_zoom, zoom_labels)
                    scaler.scale(loss_zoom).backward()
                    average_gradients(zoom_model)

                    if self.rank == 0:
                        tb_writer.add_scalar('loss_zoom_seg', loss_zoom.item(), iter)
                    scaler.step(optimizer_zoom)
                    scaler.update()

                    adjust_learning_rate(lr=self.initial_learning_rate, optimizer=optimizer,
                                         batch_size=self.train_batch_size, data_num=len(train_loader),
                                         epochs=self.num_epochs)

                    epoch_loss += loss.item()

                    if epoch == 0  or epoch==self.args.start_epoch :
                        if self.rank == 0:
                            logger.info(
                                "{}/{}, train_loss: {:.4f}".format(step, len(train_loader) // train_loader.batch_size,
                                                                   loss.item())
                            )
                        else:
                            print("{}/{}, train_loss: {:.4f}".format(step, len(train_loader) // train_loader.batch_size,
                                                                     loss.item()))


            if self.args.no_seg:
                pass
            else:
                epoch_loss /= step  # calculate mean loss over current epoch
                epoch_loss_values.append(epoch_loss)

                if self.rank == 0:
                    logger.info("epoch {} average loss: {:.4f}".format(epoch + 1, epoch_loss))
                else:
                    print("epoch {} average loss: {:.4f}".format(epoch + 1, epoch_loss))

            if self.rank == 0:
                if self.args.seg_only:
                    pass
                else:
                    grid_real_A = torchvision.utils.make_grid(
                        (transfer_container.real_A.permute(0, 1, 4, 2, 3)).squeeze().unsqueeze(dim=1))
                    tb_writer.add_image('real_A', grid_real_A, epoch)

                    grid_idt_B = torchvision.utils.make_grid(
                        (transfer_container.idt_B.permute(0, 1, 4, 2, 3)).squeeze().unsqueeze(dim=1))
                    tb_writer.add_image('idt_B', grid_idt_B, epoch)

                    grid_fake_B = torchvision.utils.make_grid(
                        (transfer_container.fake_B.permute(0, 1, 4, 2, 3)).squeeze().unsqueeze(dim=1))
                    tb_writer.add_image('fake_B', grid_fake_B, epoch)

                torch.save(model.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_model.pth"))
                torch.save(optimizer.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_model_opt.pth"))

                torch.save(zoom_model.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_zoom_model.pth"))
                torch.save(optimizer_zoom.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_model_zoom_opt.pth"))
                if not self.args.seg_only:
                    torch.save(G.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_G.pth"))
                    torch.save(D.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_D.pth"))
                    torch.save(F.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_F.pth"))
                    torch.save(optimizer_G.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_G_opt.pth"))
                    torch.save(optimizer_D.state_dict(), os.path.join(self.model_path, str(epoch) + "_epoch_D_opt.pth"))
                    torch.save(transfer_container.optimizer_F.state_dict(),
                               os.path.join(self.model_path, str(epoch) + "_epoch_F_opt.pth"))

            # # learning rate update
            # if (epoch + 1) % epochs_with_const_lr == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group["lr"] = param_group["lr"] / self.lr_divisor
            #         if self.rank == 0:
            #             logger.info(
            #                 "Dividing learning rate by {}. "
            #                 "New learning rate is: lr = {}".format(self.lr_divisor, param_group["lr"])
            #             )
            #     for param_group in optimizer_zoom.param_groups:
            #         param_group["lr"] = param_group["lr"] / self.lr_divisor

        if self.rank == 0:
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

    def run_inference(self, model,zoom_model, data_loader):
        logger = self.logger
        logger.info('Running inference...')

        # model.eval()  # activate evaluation mode of model
        for m in model.modules():
            if isinstance(m, torch.nn.SyncBatchNorm):
                m.track_running_stats = False

        for m in zoom_model.modules():
            if isinstance(m, torch.nn.SyncBatchNorm):
                m.track_running_stats = False
        dice_scores = np.zeros(len(data_loader))

        if self.model == "UNet2d5_spvPA":
            model_segmentation = lambda *args, **kwargs: model(*args, **kwargs)[0]
        else:
            model_segmentation = model

        if self.args.zoom_model == "UNet2d5_spvPA":
            zoom_model_segmentation = lambda *args, **kwargs: zoom_model(*args, **kwargs)[0]
        else:
            zoom_model_segmentation = zoom_model

        with torch.no_grad():  # turns off PyTorch's auto grad for better performance
            for i, data in enumerate(data_loader):
                logger.info("starting image {}".format(i))

                # print(data['image'].shape)
                # outputs = model_segmentation(data["image"].to(self.device))
                # print(outputs.shape)

                if self.dataset == 'both':
                    T1, T2 = data["imageT1"].to(self.device), data["imageT2"].to(
                        self.device)
                    inputs = T2
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
                    sigma_scale= 0.12,
                    mode="gaussian"
                )

                out_max, y_pred = outputs.max(dim=1, keepdim=True)
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

                # out_max, out_label = outputs.max(dim=1, keepdim=True)
                # out_label[(out_max<=0.95) & (out_label != 2)] = 0


                # export to nifti
                if self.export_inferred_segmentations:
                    logger.info(f"export to nifti...")

                    # print(data[img_dict])

                    nifti_data_matrix = np.squeeze(y_pred)[None, :]
                    if self.dataset =='both':
                        img_dict = 'imageT2_meta_dict'
                    elif self.dataset =='both_h':
                        img_dict = 'imageT2_meta_dict'
                    else:
                        img_dict = 'image_meta_dict'

                    data[img_dict]['filename_or_obj'] = os.path.join('../target_validation', 'crossmoda_' +
                                                                     data[img_dict]['filename_or_obj'][0].replace('hrT2','Label'))
                    data[img_dict]['affine'] = np.squeeze(data[img_dict]['affine'])
                    data[img_dict]['original_affine'] = np.squeeze(data[img_dict]['original_affine'])

                    print(os.path.join(self.results_folder_path, 'inferred_segmentations_nifti'))
                    saver = NiftiSaver(
                        output_dir=os.path.join(self.results_folder_path, 'inferred_segmentations_nifti'), output_postfix='')
                    saver.save(nifti_data_matrix, meta_data=data[img_dict])
                    # save_label = nib.Nifti1Image((nifti_data_matrix.cpu().numpy()).astype(np.int16), np.squeeze(data[img_dict]['affine']))
                    # print(save_label.shape)
                    # nib.save(save_label, os.path.join(self.results_folder_path, 'inferred_segmentations_nifti',os.path.basename(data[img_dict]['filename_or_obj'][0].replace('hrT2','Label'))) )
                    print(data[img_dict]['filename_or_obj'], np.unique(nifti_data_matrix.cpu().numpy()))

    def run_transfer(self, model, data_loader):
        logger = self.logger
        logger.info('Running inference...')

        # model.eval()  # activate evaluation mode of model
        for m in model.modules():
            if isinstance(m, torch.nn.SyncBatchNorm):
                m.track_running_stats = False

        with torch.no_grad():  # turns off PyTorch's auto grad for better performance
            for i, data in enumerate(data_loader):
                logger.info("starting image {}".format(i))

                # print(data['image'].shape)
                # outputs = model_segmentation(data["image"].to(self.device))
                # print(outputs.shape)

                if self.dataset == 'both_sep':
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
                    predictor=model,
                    overlap=0.5,
                    sigma_scale= 0.12,
                    mode="gaussian"
                )


                # outputs = model_segmentation(inputs)
                print(outputs.max(),outputs.min())
                outputs = ((outputs + 1.0) * self.args.intensity/2.0).detach().cpu().numpy()

                # dice_score = self.compute_dice_score(outputs, data["label"].to(self.device))
                # dice_scores[i] = dice_score.item()
                #
                # logger.info(f"dice_score = {dice_score.item()}")

                # export to nifti
                if self.export_inferred_segmentations:
                    logger.info(f"export to nifti...")

                    # print(data[img_dict])

                    nifti_data_matrix = (np.squeeze(outputs).astype(np.int16))[None, :]
                    if self.dataset =='both_sep':
                        img_dict = 'imageT2_meta_dict'
                    elif self.dataset =='both_h':
                        img_dict = 'imageT2_meta_dict'
                    else:
                        img_dict = 'image_meta_dict'

                    data[img_dict]['filename_or_obj'] = os.path.join('../target_validation', 'crossmoda_' +
                                                                     data[img_dict]['filename_or_obj'][0].replace('ceT1','hrT2'))
                    data[img_dict]['affine'] = np.squeeze(data[img_dict]['affine'])
                    data[img_dict]['original_affine'] = np.squeeze(
                        data[img_dict]['original_affine'])

                    saver = NiftiSaver(
                        output_dir=os.path.join(self.results_folder_path, 'transfered_hrT2'), output_postfix='')
                    saver.save(nifti_data_matrix, meta_data=data[img_dict])
                    print(data[img_dict]['filename_or_obj'])