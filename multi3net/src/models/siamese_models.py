# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

from utils.trainer import tensor_to_variable

from models.unet_parts import * 

class SiamUnet_conc(nn.Module):
    """SiamUnet_conc segmentation network."""

    def __init__(self, n_channels, n_classes):
        super(SiamUnet_conc, self).__init__()
	
        self.n_classes = n_classes
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up_skip_conc(512, 256)
        self.up2 = up_skip_conc(256, 128)
        self.up3 = up_skip_conc(128, 64)
        self.up4 = up_skip_conc(64, 64)
        self.outc = outconv(64, n_classes)

       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
               m.weight.data.normal_(0, math.sqrt(2. / n))
           elif isinstance(m, nn.BatchNorm2d):
               m.weight.data.fill_(1)
               m.bias.data.zero_()


    def forward(self, input):
        post = tensor_to_variable(input['vhr_post'])
        pre = tensor_to_variable(input['vhr_pre'])

        pre_x1 = self.inc(pre)
        pre_x2 = self.down1(pre_x1)
        pre_x3 = self.down2(pre_x2)
        pre_x4 = self.down3(pre_x3)
        pre_x5 = self.down4(pre_x4)

        post_x1 = self.inc(post)
        post_x2 = self.down1(post_x1)
        post_x3 = self.down2(post_x2)
        post_x4 = self.down3(post_x3)
        post_x5 = self.down4(post_x4)

        x = self.up1(post_x5, torch.cat([pre_x4, post_x4], dim=1))
        x = self.up2(x, torch.cat([pre_x3, post_x3], dim=1))
        x = self.up3(x, torch.cat([pre_x2, post_x2], dim=1))
        x = self.up4(x, torch.cat([pre_x1, post_x1], dim=1))
        x = self.outc(x)

    	class_idx = x.size().index(self.n_classes)
        return F.log_softmax(x, dim=class_idx)

class up_skip_conc(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, halved_input=True):
        super(up_skip_conc, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=stride)
        self.conv = double_conv(int(2.5 * in_ch), out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class SiamUnet_diff(nn.Module):
    """SiamUnet_conc segmentation network."""

    def __init__(self, n_channels, n_classes):
        super(SiamUnet_diff, self).__init__()
	
        self.n_classes = n_classes
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up_skip(1024, 256)
        self.up2 = up_skip(512, 128)
        self.up3 = up_skip(256, 64)
        self.up4 = up_skip(128, 64)
        self.outc = outconv(64, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input):
        post = tensor_to_variable(input['vhr_post'])
        pre = tensor_to_variable(input['vhr_pre'])

        pre_x1 = self.inc(pre)
        pre_x2 = self.down1(pre_x1)
        pre_x3 = self.down2(pre_x2)
        pre_x4 = self.down3(pre_x3)
        pre_x5 = self.down4(pre_x4)

        post_x1 = self.inc(post)
        post_x2 = self.down1(post_x1)
        post_x3 = self.down2(post_x2)
        post_x4 = self.down3(post_x3)
        post_x5 = self.down4(post_x4)

        x = self.up1(post_x5, post_x4 - pre_x4)
        x = self.up2(x, post_x3 - pre_x3)
        x = self.up3(x, post_x2 - pre_x2)
        x = self.up4(x, post_x1 - pre_x1)
        x = self.outc(x)

    	class_idx = x.size().index(self.n_classes)
        return F.log_softmax(x, dim=class_idx)

    
