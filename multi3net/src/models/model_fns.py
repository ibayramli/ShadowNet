from models.pspnet.resnet import *
from models.pspnet.psp_net import PSPNet
from models.fusenet_model import UNet
from models.fusenet_model import UNet_PSP


def pspnet_10m():
    model = PSPNet(resnet34(input_channels=3, pretrained=False), psp_size=512)
    return model

def pspnet_10m_pre_post():
    model = PSPNet(resnet34(input_channels=6, pretrained=False), psp_size=512)
    return model

def unet_basic_vhr():
    model = UNet(3, 2)
    return model

def unet_psp():
    model = UNet_PSP(3, 2)
