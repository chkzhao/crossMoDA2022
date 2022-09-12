import math
# from utils.GIN import GIN
import random

try:
    from scipy.special import comb
except:
    from scipy.misc import comb

import monai
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
# from model.cut_model import PatchNCELoss
import numpy as np
from monai.config import DtypeLike, KeysCollection
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import convert_data_type
from monai.data.image_reader import ITKReader
import torch.nn as nn
from collections import OrderedDict
from monai.transforms import MapTransform

class InstanceNormalizeIntensityd(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,
            dtype: DtypeLike = np.float32
    ) -> None:
        super(InstanceNormalizeIntensityd, self).__init__(keys, allow_missing_keys)
        self.dtype = dtype

    def instanceNormalize(self, image):
        # print(image.shape)

        img_9999_percent = np.percentile(image, 99.99)

        image[image < 0] = 0
        image[image > img_9999_percent] = img_9999_percent

        # max_value = torch.max(image) / 2
        # new_image = (image - max_value) / max_value
        # max_value = torch.max(image)
        # new_image = image / max_value
        max_value = torch.max(image) / 2
        new_image = (image - max_value) / max_value

        return new_image

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            if key in data:
                d[key] = self.instanceNormalize(d[key])
        return d

