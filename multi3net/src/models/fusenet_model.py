# full assembly of the sub-parts to form the complete net

from .unet_parts import *

import math

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from six import text_type
from six import binary_type
from collections import OrderedDict

from models.damage.psp_net_fusion import AttentionNetSimple
from models.pspnet.psp_net import PSPModule

from utils.trainer import tensor_to_variable



class FuseNet(nn.Module):
    def __init__(self, n_classes, channel_dict, fusion, tile_size):
        super(FuseNet, self).__init__()

        self.channel_dict = channel_dict
        self.n_conv_channels = 64
        self.n_classes = n_classes
        self.fusion = fusion
        self.tile_size = tile_size

        # self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        self.softmax = nn.LogSoftmax()

        if self.fusion == 'SVM_base':
            self.S2UpNet10 = S2UpNet10()
            S2_channels = self.S2UpNet10.output_channels

            self.UNet = UNet(S2_channels + self.channel_dict["img10"], n_classes)

        if self.fusion == 'exp':
            self.VHRDownNet5 = VHRDownNet5()
            vhr_channels = self.VHRDownNet5.output_channels

            self.UNet = UNet(vhr_channels, n_classes)

        if self.fusion == 'onlySenTo5m':  # Sen-2 optical up to 5m, fused in Unet with SAR
            n_channels = self.channel_dict["img10"] + self.channel_dict["sar"]
            self.UNet = UNet(n_channels, n_classes)

        if self.fusion == 'BaselineLastWeek':  # All (or subset) Sen-1 and Sen-2 modalities to 10m resolution
            n_channels = self.channel_dict["img10"] + self.channel_dict["img20"] + self.channel_dict["img60"] + self.channel_dict["sar"]
            self.UNet = UNet(n_channels, n_classes)

        if self.fusion == 'BaselineVHR':  # Unet on VHR only (upsampling targets)
            self.UNet = UNet(self.channel_dict["vhr"], n_classes)

        if self.fusion == 'FuseAt5Conv':  # VHR conv to 5m res, S2 20m and 60m res conv to 5m, fuse at 5m
            self.VHRDownNet5 = VHRDownNet5()
            vhr_channels = self.VHRDownNet5.output_channels

            self.S2UpNet5 = S2UpNet5()
            S2_channels = self.S2UpNet5.output_channels

            in_channels = S2_channels + vhr_channels + self.channel_dict["sar"]

            self.UNet = UNet(in_channels, n_classes)

        if self.fusion == 'FuseAt10Conv':  # VHR conv to 10m res, S2 20m and 60m res conv to 10m, fuse at 10m
            self.VHRDownNet10 = VHRDownNet10()
            vhr_channels = self.VHRDownNet10.output_channels

            self.S2UpNet10 = S2UpNet10()
            S2_channels = self.S2UpNet10.output_channels

            in_channels = S2_channels + vhr_channels + self.channel_dict["img10"] + self.channel_dict["sar"]

            self.UNet = UNet(in_channels, n_classes)

        if self.fusion == 'FuseAt10ConvNoVHR':  # S2 20m and 60m res conv to 10m, fuse at 10m

            self.S2UpNet10 = S2UpNet10()
            S2_channels = self.S2UpNet10.output_channels

            in_channels = S2_channels + self.channel_dict["img10"] + self.channel_dict["sar"]

            self.UNet = UNet(in_channels, n_classes)

        if self.fusion == 'FuseAt10Upsample': # VHR downsample to 10m res, S2 20m and 60m res conv to 10m, fuse at 10m
            self.VHRDownNet10 = VHRDownNet10()
            vhr_channels = self.VHRDownNet10.output_channels

            in_channels = vhr_channels + self.channel_dict["img10"] + self.channel_dict["img20"] + self.channel_dict["img60"] + self.channel_dict["sar"]

            self.UNet = UNet(in_channels, n_classes)


