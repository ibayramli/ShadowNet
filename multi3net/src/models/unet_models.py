# full assembly of the sub-parts to form the complete net

from unet_parts import *

import math
import copy

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from six import text_type
from six import binary_type
from collections import OrderedDict

from models.damage.psp_net_fusion import AttentionNetSimple
from models.pspnet.psp_net import PSPModule

from utils import resume
from utils.trainer import tensor_to_variable


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
	
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
        x = tensor_to_variable(input['vhr'])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
 
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)


    	class_idx = x.size().index(self.n_classes)
        return F.log_softmax(x, dim=class_idx)

class UNet_PSP(nn.Module):
    def __init__(self, n_channels, n_classes, psp_sizes=(1, 2, 3, 6)):
        super(UNet_PSP, self).__init__()
	
        self.n_classes = n_classes
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.psp = PSPModule(512, 1024, psp_sizes)
        

        self.up1 = up_skip(1024, 512, halved_input=False)
        self.up2 = up_skip(512, 256, halved_input=False)
        self.up3 = up_skip(256, 128, halved_input=False)
        self.up4 = up_skip(128, 64, halved_input=False)
        self.outc = outconv(64, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input):
        x = tensor_to_variable(input['vhr'])
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.psp(x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

    	class_idx = x.size().index(self.n_classes)
        return F.log_softmax(x, dim=class_idx)


class FUseNet(nn.Module):
    def __init__(self, n_channels, n_classes, post_encoder):
        super(FUseNet, self).__init__()
	
        self.n_classes = n_classes
        self.post_encoder = post_encoder

        self.inc = inconv(n_channels, 64)
	
        self.conv0 = nn.Conv2d(128, 64, kernel_size=1)

        self.down1 = down(64, 128)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)

        self.down2 = down(128, 256)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=1)

        self.down3 = down(256, 512)
        self.conv3 = nn.Conv2d(1024, 512, kernel_size=1)

        self.down4 = down(512, 512)
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1), 
            double_conv(512, 1024)
        )

        self.up1 = up_skip(1024, 512, halved_input=False)
        self.up2 = up_skip(512, 256, halved_input=False)
        self.up3 = up_skip(256, 128, halved_input=False)
        self.up4 = up_skip(128, 64, halved_input=False)
        self.outc = outconv(64, n_classes)

        for name, module in self.named_children():
            if name != 'post_encoder':        
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()


    def get_activations(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook


    def forward(self, input):
        post = tensor_to_variable(input['vhr_post'])
        pre = tensor_to_variable(input['vhr_pre'])
        
	with torch.no_grad():
            _, activations = self.post_encoder(post)

        x0 = self.inc(pre)
        x0 = torch.cat([x0, activations['inc']], dim=1)
        x0 = self.conv0(x0)

        x1 = self.down1(x0)
        x1 = torch.cat([x1, activations['down1']], dim=1)
        x1 = self.conv1(x1)

        x2 = self.down2(x1)
        x2 = torch.cat([x2, activations['down2']], dim=1)
        x2 = self.conv2(x2)

        x3 = self.down3(x2)
        x3 = torch.cat([x3, activations['down3']], dim=1)
        x3 = self.conv3(x3)

        x4 = self.down4(x3)
        x4 = torch.cat([x4, activations['down4']], dim=1)
        x4 = self.conv4(x4)
 	
        x = self.up1(x4, activations['down3'])
        x = self.up2(x, activations['down2'])
        x = self.up3(x, activations['down1'])
        x = self.up4(x, activations['inc'])
        x = self.outc(x)

        return F.log_softmax(x, dim=1)

class UNet_Encoder(nn.Module):
    def __init__(self, n_channels):
        super(UNet_Encoder, self).__init__()
	
        self.inc = inconv(n_channels, 64)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

	activations = {
            'inc': x1,
            'down1': x2,
            'down2': x3,
            'down3': x4,
            'down4': x5
        }

        return (x5, activations)

def unet_encoder(in_channels, out_channels, encoder_path, encoder_depth=5):
    unet_enc = UNet_Encoder(in_channels)
    encoder_state = unet_enc.state_dict()
    unet_state = torch.load(encoder_path)['model_state']
    
    # if weights were wrapped in nn.DataParallel remove .module form state keys
    if 'module' in unet_state.keys()[0]:
        for key in unet_state.keys():
            pre, post = key.split('.', 1)
            weights = unet_state.pop(key)
            if post in encoder_state.keys():
                unet_state[post] = weights

    unet_enc.load_state_dict(unet_state)
    return unet_enc


# Implementation of a model from https://arxiv.org/pdf/1810.08462.pdf
class FC_EF(nn.Module):    
    def __init__(self, n_channels, n_classes):
        super(FC_EF, self).__init__()
	
        self.n_classes = n_classes
        self.inc = inconv(2*n_channels, 64)
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
        x = torch.cat([pre, post])
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
 
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)


    	class_idx = x.size().index(self.n_classes)
        return F.log_softmax(x, dim=class_idx)

