from models.pspnet.resnet import *
from models.pspnet.psp_net import PSPNet
from models.unet_models import *
from models.siamese_models import *

def pspnet_10m():
    model = PSPNet(resnet34(input_channels=3, pretrained=False), psp_size=512)
    return model

def pspnet_upsample():
    model = PSPNet(resnet34(input_channels=3, pretrained=False), psp_size=512, upsample_factors=[2, 2, 2])
    return model

def pspnet_10m_pre_post():
    model = PSPNet(resnet34(input_channels=6, pretrained=False), psp_size=512)
    return model

def unet_basic_vhr():
    model = UNet(3, 2)
    return model

def unet_psp():
    model = UNet_PSP(3, 2)
    return model

def fusenet(encoder_path):
    encoder = unet_encoder(3, 2, encoder_path)
    model = FUseNet(3, 2, encoder)
    return model
    
def additive_fusenet(encoder_path):
    encoder = unet_encoder(3, 2, encoder_path)
    model = AdditiveFUseNet(3, 2, encoder)
    return model

def fc_ef():
    model = FC_EF(3, 2)
    return model

def siam_unet_conc():
    model = SiamUnet_conc(3, 2)
    return model 

def siam_unet_diff():
    model = SiamUnet_diff(3, 2)
    return model



