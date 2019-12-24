from torchsummary import summary
from models.pspnet.psp_net import pspnet_10m

summary(pspnet_10m(), input_size=(3, 1024, 1024))

