# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn

from params.networks.blocks.convolutions import Convolution, ResidualUnit
from ..blocks.attentionblock import AttentionBlock1, AttentionBlock2
from monai.networks.layers.factories import Norm, Act
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import export
from monai.utils.aliases import alias
import torch.nn.functional as F
from .deeplabv3_3d import DeepLabV3_3D

num_classes = 10            # Number of classes. (= number of output channel)
input_channels = 3          # Number of input channel
resnet = 'resnet18_os16'    # Base resnet architecture ('resnet18_os16', 'resnet34_os16', 'resnet50_os16', 'resnet101_os16', 'resnet152_os16', 'resnet18_os8', 'resnet34_os18')
last_activation = 'softmax' # 'softmax', 'sigmoid' or None

model = DeepLabV3_3D(num_classes = num_classes, input_channels = input_channels, resnet = resnet, last_activation = last_activation)


def center_crop(img, size):
    H, W, D = img.shape[2], img.shape[3], img.shape[4]

    start_h = int((H - size[0])/2.0)
    start_w = int((W - size[1]) / 2.0)
    start_d = int((D - size[2]) / 2.0)

    new_img = img[:, :, start_h:start_h+size[0], start_w:start_w+size[1], start_d:start_d+size[2]]
    return new_img





class Down_Net(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.

    Args:
        dimensions: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        subunits: number of convolutions. Defaults to 2.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.
            When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).
            The value of dropout_dim should be no no larger than the value of dimensions.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.

    See also:

        :py:class:`monai.networks.blocks.Convolution`

    """

    def __init__(
        self, in_channels, out_channels, kernel_size,num_res_units,act,norm,dropout, dimensions) -> None:
        super().__init__()

        if num_res_units > 0:
            self.net = ResidualUnit(
                in_channels = in_channels,
                out_channels = out_channels,
                dimensions = dimensions,
                strides=1,
                kernel_size=kernel_size,
                subunits=num_res_units,
                act=act,
                norm=norm,
                dropout=dropout,
            )
        else:
            self.net = Convolution(
                in_channels=in_channels,
                out_channels=out_channels,
                dimensions=dimensions,
                strides=1,
                kernel_size=kernel_size,
                act=act,
                norm=norm,
                dropout=dropout,
            )

    def forward(self, input):

        return self.net(input)

class Down_sample(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.

    Args:
        dimensions: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        subunits: number of convolutions. Defaults to 2.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.
            When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).
            The value of dropout_dim should be no no larger than the value of dimensions.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.

    See also:

        :py:class:`monai.networks.blocks.Convolution`

    """

    def __init__(
        self, dimensions, in_channels, out_channels, kernel_size,strides,act,norm,dropout) -> None:
        super().__init__()

        self.net = Convolution(
                dimensions = dimensions,
                in_channels = in_channels,
                out_channels= out_channels,
                strides = strides,
                kernel_size = kernel_size,
                act = act,
                norm = norm,
                dropout = dropout,
                is_transposed=False,
            )

    def forward(self, input):

        return self.net(input)

def conv_block_3d_assp(in_dim, out_dim, dilation):
    model = nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=[3,3,1], stride=1, padding=dilation, dilation=dilation),
    )
    return model

class ASSP(nn.Module):
    def __init__(
        self, num_filter) -> None:
        super().__init__()

        self.assp1 = conv_block_3d_assp(num_filter, num_filter, 1)
        self.assp2 = conv_block_3d_assp(num_filter, num_filter, 2)
        self.assp3 = conv_block_3d_assp(num_filter, num_filter, 4)

    def forward(self, input):
        down_3_assp1 = self.assp1(input)
        down_3_assp2 = self.assp2(input)
        down_3_assp3 = self.assp3(input)
        out_put = torch.cat([down_3_assp1, down_3_assp2, down_3_assp3], dim=1)

        return out_put

@export("monai.networks.nets")
@alias("Unet2d5_spvPA")
@export("monai.networks.nets")
@alias("Unet2d5_spvPA")
class UNet2d5_spvPA(nn.Module):
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        channels,
        strides,
        kernel_sizes,
        sample_kernel_sizes,
        num_res_units=0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0,
        attention_module=True,
        zoom_in = False
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == (len(strides)) + 1 == len(sample_kernel_sizes) + 1
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.sample_kernel_sizes = sample_kernel_sizes
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.attention_module = attention_module
        self.att_maps = []
        self.zoom_in = zoom_in

        self.upsample1 = nn.Upsample(scale_factor=(1.5, 1.5, 1), mode='trilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=(3, 3, 1), mode='trilinear', align_corners=True)


        def _create_block(inc, outc, channels, strides, kernel_sizes, sample_kernel_sizes, is_top):
            """
            Builds the UNet2d5_spvPA structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            """
            c = channels[0]
            s = strides[0]
            k = kernel_sizes[0]
            sk = sample_kernel_sizes[0]

            # create layer in downsampling path
            down = self._get_down_layer(in_channels=inc, out_channels=c, kernel_size=k)
            downsample = self._get_downsample_layer(in_channels=c, out_channels=c, strides=s, kernel_size=sk)

            if len(channels) > 2:
                # continue recursion down
                subblock = _create_block(
                    c, channels[1], channels[1:], strides[1:], kernel_sizes[1:], sample_kernel_sizes[1:], is_top=False
                )
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(
                    in_channels=c,
                    out_channels=channels[1],
                    kernel_size=kernel_sizes[1],
                )

            upsample = self._get_upsample_layer(in_channels=channels[1], out_channels=c, strides=s, up_kernel_size=sk)
            subblock_with_resampling = nn.Sequential(downsample, subblock, upsample)

            # create layer in upsampling path
            up = self._get_up_layer(in_channels=2 * c, out_channels=outc, kernel_size=k, is_top=is_top)

            return nn.Sequential(down, SkipConnection(subblock_with_resampling), up)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, self.kernel_sizes, self.sample_kernel_sizes, True
        )

        # register forward hooks on all Attentionblock1 modules, to save the attention maps
        if self.attention_module:
            for layer in self.model.modules():
                if type(layer) == AttentionBlock1:
                    layer.register_forward_hook(self.hook_save_attention_map)

    def hook_save_attention_map(self, module, inp, outp):
        if len(self.att_maps) == len(self.channels):
            self.att_maps = []
        self.att_maps.append(outp[0])  # get first element of output (Attentionblock1 returns (att, x) )

    def _get_att_layer(self, in_channels, out_channels, kernel_size):
        att1 = AttentionBlock1(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        att2 = AttentionBlock2(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        return nn.Sequential(att1, att2)

    def _get_down_layer(self, in_channels, out_channels, kernel_size):
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )

    def _get_downsample_layer(self, in_channels, out_channels, strides, kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=False,
        )
        return conv

    def _get_bottom_layer(self, in_channels, out_channels, kernel_size):
        conv = self._get_down_layer(in_channels, out_channels, kernel_size)
        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)
            return nn.Sequential(att_layer, conv)
        else:
            return conv

    def _get_upsample_layer(self, in_channels, out_channels, strides, up_kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            up_kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=True,
        )
        return conv

    def _get_up_layer(self, in_channels, out_channels, kernel_size, is_top):

        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=1,  # why not self.num_res_units?
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )

        if self.attention_module and self.num_res_units > 0:
            return nn.Sequential(att_layer, ru)
        elif self.attention_module and not self.num_res_units > 0:
            return att_layer
        elif self.num_res_units > 0 and not self.attention_module:
            return ru
        elif not self.attention_module and not self.num_res_units > 0:
            return nn.Identity
        else:
            raise NotImplementedError

    def forward(self, input):
        x = self.model(input)
        return x, self.att_maps


Unet2d5_spvPA = unet2d5_spvPA = UNet2d5_spvPA

class UNet2d5_spvPA_zoom(nn.Module):
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        channels,
        strides,
        kernel_sizes,
        sample_kernel_sizes,
        num_res_units=0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0,
        attention_module=True,
        zoom_in = False
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == (len(strides)) + 1 == len(sample_kernel_sizes) + 1
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.sample_kernel_sizes = sample_kernel_sizes
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.attention_module = attention_module
        self.att_maps = []
        self.zoom_in = zoom_in

        self.upsample1 = nn.Upsample(scale_factor=(1.5, 1.5, 1), mode='trilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=(3, 3, 1), mode='trilinear', align_corners=True)


        def _create_block(inc, outc, channels, strides, kernel_sizes, sample_kernel_sizes, is_top):
            """
            Builds the UNet2d5_spvPA structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            """
            c = channels[0]
            s = strides[0]
            k = kernel_sizes[0]
            sk = sample_kernel_sizes[0]

            # create layer in downsampling path
            down = self._get_down_layer(in_channels=inc, out_channels=c, kernel_size=k)
            downsample = self._get_downsample_layer(in_channels=c, out_channels=c, strides=s, kernel_size=sk)

            if len(channels) > 2:
                # continue recursion down
                subblock = _create_block(
                    c, channels[1], channels[1:], strides[1:], kernel_sizes[1:], sample_kernel_sizes[1:], is_top=False
                )
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(
                    in_channels=c,
                    out_channels=channels[1],
                    kernel_size=kernel_sizes[1],
                )

            upsample = self._get_upsample_layer(in_channels=channels[1], out_channels=c, strides=s, up_kernel_size=sk)
            subblock_with_resampling = nn.Sequential(downsample, subblock, upsample)

            # create layer in upsampling path
            up = self._get_up_layer(in_channels=2 * c, out_channels=outc, kernel_size=k, is_top=is_top)

            return nn.Sequential(down, SkipConnection(subblock_with_resampling), up)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, self.kernel_sizes, self.sample_kernel_sizes, True
        )

        # register forward hooks on all Attentionblock1 modules, to save the attention maps
        if self.attention_module:
            for layer in self.model.modules():
                if type(layer) == AttentionBlock1:
                    layer.register_forward_hook(self.hook_save_attention_map)

    def hook_save_attention_map(self, module, inp, outp):
        if len(self.att_maps) == len(self.channels):
            self.att_maps = []
        self.att_maps.append(outp[0])  # get first element of output (Attentionblock1 returns (att, x) )

    def _get_att_layer(self, in_channels, out_channels, kernel_size):
        att1 = AttentionBlock1(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        att2 = AttentionBlock2(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        return nn.Sequential(att1, att2)

    def _get_down_layer(self, in_channels, out_channels, kernel_size):
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )

    def _get_downsample_layer(self, in_channels, out_channels, strides, kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=False,
        )
        return conv

    def _get_bottom_layer(self, in_channels, out_channels, kernel_size):
        conv = self._get_down_layer(in_channels, out_channels, kernel_size)
        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)
            return nn.Sequential(att_layer, conv)
        else:
            return conv

    def _get_upsample_layer(self, in_channels, out_channels, strides, up_kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            up_kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=True,
        )
        return conv

    def _get_up_layer(self, in_channels, out_channels, kernel_size, is_top):

        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=1,  # why not self.num_res_units?
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )

        if self.attention_module and self.num_res_units > 0:
            return nn.Sequential(att_layer, ru)
        elif self.attention_module and not self.num_res_units > 0:
            return att_layer
        elif self.num_res_units > 0 and not self.attention_module:
            return ru
        elif not self.attention_module and not self.num_res_units > 0:
            return nn.Identity
        else:
            raise NotImplementedError

    def forward(self, input):

        input = F.interpolate(input, scale_factor=2, mode='trilinear', align_corners=True)
        x = self.model(input)
        x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True)
        att_maps = []
        for att in self.att_maps:
            att_maps.append(F.interpolate(att, scale_factor=0.5, mode='trilinear', align_corners=True))
        return x, att_maps

class UNet2d5_spvPA_T(nn.Module):
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        channels,
        strides,
        kernel_sizes,
        sample_kernel_sizes,
        num_res_units=0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0,
        attention_module=True,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == (len(strides)) + 1 == len(sample_kernel_sizes) + 1
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.sample_kernel_sizes = sample_kernel_sizes
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.attention_module = attention_module
        self.feature_maps = []

        def _create_block(inc, outc, channels, strides, kernel_sizes, sample_kernel_sizes, is_top):
            """
            Builds the UNet2d5_spvPA structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            """
            c = channels[0]
            s = strides[0]
            k = kernel_sizes[0]
            sk = sample_kernel_sizes[0]

            # create layer in downsampling path
            down = Down_Net(in_channels=inc, out_channels=c, kernel_size=k, num_res_units = self.num_res_units,
                            act=self.act,norm=self.norm,dropout=self.dropout,dimensions=self.dimensions)
            downsample = Down_sample(dimensions=self.dimensions, in_channels=c, out_channels=c, strides=s, kernel_size=sk, act = self.act,norm=self.norm,dropout=self.dropout)

            if len(channels) > 2:
                # continue recursion down
                subblock = _create_block(
                    c, channels[1], channels[1:], strides[1:], kernel_sizes[1:], sample_kernel_sizes[1:], is_top=False
                )
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(
                    in_channels=c,
                    out_channels=channels[1],
                    kernel_size=kernel_sizes[1],
                )

            upsample = self._get_upsample_layer(in_channels=channels[1], out_channels=c, strides=s, up_kernel_size=sk)
            subblock_with_resampling = nn.Sequential(downsample, subblock, upsample)

            # create layer in upsampling path
            up = self._get_up_layer(in_channels=2 * c, out_channels=outc, kernel_size=k, is_top=is_top)

            return nn.Sequential(down, SkipConnection(subblock_with_resampling), up)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, self.kernel_sizes, self.sample_kernel_sizes, True
        )

        self.layer_num = 3
        i = 0
        if self.attention_module:
            for layer in self.model.modules():
                if type(layer) == Down_Net and i < self.layer_num:
                    layer.register_forward_hook(self.hook_save_feature_map)
                    i += 1

    def hook_save_feature_map(self, module, inp, outp):
        if len(self.feature_maps) == self.layer_num + 1:
            self.feature_maps = []
        self.feature_maps.append(outp)

    def _get_att_layer(self, in_channels, out_channels, kernel_size):
        att1 = AttentionBlock1(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        att2 = AttentionBlock2(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        return nn.Sequential(att1, att2)

    def _get_down_layer(self, in_channels, out_channels, kernel_size):
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )

    def _get_bottom_layer(self, in_channels, out_channels, kernel_size):
        conv = self._get_down_layer(in_channels, out_channels, kernel_size)
        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)
            return nn.Sequential(att_layer, conv)
        else:
            return conv

    def _get_upsample_layer(self, in_channels, out_channels, strides, up_kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            up_kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=True,
        )
        return conv

    def _get_up_layer(self, in_channels, out_channels, kernel_size, is_top):

        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=1,  # why not self.num_res_units?
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )

        if self.attention_module and self.num_res_units > 0:
            return nn.Sequential(att_layer, ru)
        elif self.attention_module and not self.num_res_units > 0:
            return att_layer
        elif self.num_res_units > 0 and not self.attention_module:
            return ru
        elif not self.attention_module and not self.num_res_units > 0:
            return nn.Identity
        else:
            raise NotImplementedError

    def forward(self, input, encode_only = False):
        if encode_only:
            self.model(input)
            self.feature_maps.append(input)
            return self.feature_maps
        else:
            x = nn.Tanh()(self.model(input))
            self.feature_maps.append(input)
            return x

Unet2d5_spvPA = unet2d5_spvPA = UNet2d5_spvPA
