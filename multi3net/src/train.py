from __future__ import print_function, division

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import argparse

from utils import resume

from utils.dataloader import train_xbd_data_loader
from utils.dataloader import val_xbd_data_loader

from utils.trainer import Trainer

from utils.losses import lovasz_softmax, DiceLoss

from models.low_res_seg.keep_dimension import input_keep_res_net_34_s1_all
from models.low_res_seg.keep_dimension import input_keep_res_net_34_s2_all 
from models.low_res_seg.keep_dimension import input_keep_res_net_34_vhr_all

from models.pspnet.pspnet_sentinel import psp34_sentinel1_and_sentinel2
from models.pspnet.pspnet_fused import pspnet_fused_s2_10m
from models.pspnet.pspnet_fused import pspnet_fused_s1_10m
from models.pspnet.pspnet_fused_all import pspnet_fused_s1s2_10m


from models.model_fns import *

TRAINDATA_ENVIRONMENT_VARIABLE="TRAINDATA_PATH"
VALIDATA_ENVIRONMENT_VARIABLE="VALIDATA_PATH"

def main(
        batch_size,
        nworkers,
        outdir,
        num_epochs,
        snapshot,
        finetune,
        lr,
        lradapt,
        experiment,
        labelimage,
        smoketest=False,
        trainpath=None,
        validpath=None
    ):

    np.random.seed(0)
    torch.manual_seed(0)

    # Visdom environment
    visdom_environment = 'vhr_' + experiment + '_' + labelimage.replace(".tif", "")
    outdir = os.path.join(outdir, visdom_environment)

    if validpath is None:
        validpath = os.environ[VALIDATA_ENVIRONMENT_VARIABLE]
    if trainpath is None:
        trainpath = os.environ[TRAINDATA_ENVIRONMENT_VARIABLE]

    train = train_xbd_data_loader(trainpath, batch_size=batch_size, shuffle=True, mode='train', num_workers=nworkers, experiment=experiment)
    val = val_xbd_data_loader(validpath, batch_size=batch_size, shuffle=True, mode='val',  num_workers=nworkers, experiment=experiment) 


    if experiment == "pre_post":
	    network = siam_unet_diff()
#        network = fusenet('../results/predictions_single_unet_basic_weight_3/vhr_buildings10m/epoch_20_classes_02.pth')
    elif experiment == "pre" or experiment == 'post':
        network = unet_basic_vhr()
    else:
        raise ValueError("Please insert a valid experiment id. Valid experiments are 'pre', 'post', 'per_post'")
	
    if torch.cuda.device_count() > 1:
	network = nn.DataParallel(network)
    if torch.cuda.is_available():
        network = network.cuda()

    if finetune or snapshot:
        resume(finetune or snapshot, network, None)

    optimizer = optim.Adam(network.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=lradapt)

    if snapshot:
        state = resume(snapshot, None, optimizer)
        train.iterations = state['iteration']

    #loss = DiceLoss(weight=torch.tensor([20]), sigmoid_normalization=False)
    class_weights = torch.tensor([1., 3.])
    loss = nn.NLLLoss2d(weight=class_weights)
    #losis = lovasz_softmax
    if torch.cuda.is_available():
        loss = loss.cuda()

    trainer = Trainer(
        network, optimizer, scheduler, loss, train, val,
        outdir, visdom_environment, smoketest
    )
    trainer.train(num_epochs, start_epoch=0)
    

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, )
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='define size of mini-batch', )
    parser.add_argument('-w', '--workers', default=1, type=int, help='number of dataloader workers, i.e., multi-threaded processes')
    parser.add_argument('-o', '--outdir', default='/tmp', type=str, help='output directors (defaults to /tmp)', )
    parser.add_argument('-e', '--num-epochs', default=10, type=int, help='number of epochs', )
    parser.add_argument('-r', '--resume', default='', type=str, help='snapshot path to pretrained models with epoch and optimizer information', )
    parser.add_argument('-f', '--finetune', default='', type=str, help='finetune path to weights only')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('-x', '--experiment', default="vhrs1s2", type=str, help="experiment name. Possible values: 'vhr', 's1', 's2', 'vhrs1, 'vhrs2', 'vhrs1s2' (default)")
    parser.add_argument('-a', '--lradapt', default=1, type=float, help='decrease learning rate incrementally. Defaults to 1: no decrease')
    parser.add_argument('-k', '--weight', default=1, type=int, help='weight parameter class 1 more than 0 (background). Defaults to one')
    parser.add_argument('-l', '--labelimage', default='buildings10m.tif', type=str, help="name of the label image in the dataset. either 'buildings{10,2,1}m.tif' or 'flooded{10,2,1}m.tif'")
    parser.add_argument('--trainpath', default=None, type=str, help='path to training data (if blank uses environment variable TRAINDATA_PATH)')
    parser.add_argument('--validpath', default=None, type=str, help='path to validation data (if blank uses environment variable VALIDATA_PATH)')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    try:
        main(
            args.batch_size,
            args.workers,
            args.outdir,
            args.num_epochs,
            args.resume,
            args.finetune,
            args.lr,
            args.lradapt,
            args.experiment,
            args.labelimage,
            args.trainpath,
            args.validpath
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
