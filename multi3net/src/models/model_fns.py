from models.pspnet.resnet import *
from models.pspnet.psp_net import PSPNet
from models.fusenet_model import UNet as UNetBasic 
from models.unet_model import UNet


def pspnet_10m():
    model = PSPNet(resnet34(input_channels=3, pretrained=False), psp_size=512)
    return model

def pspnet_10m_pre_post():
    model = PSPNet(resnet34(input_channels=6, pretrained=False), psp_size=512)
    return model

def unet_basic_vhr():
    model = UNetBasic(3, 2)
    return model

def unet_encoded_vhr():
    channel_dict = {'vhr' : 3}

    model = UNet(2, channel_dict, 'vhr')
    return model
