from __future__ import print_function, division

from torch import optim
from utils import resume
from utils.trainer import Trainer

import utils.classmetric as classmetric
import torch.nn as nn
import torch
from torch.utils import data
from utils.image import tiff_to_nd_array
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import pwd
import csv
import pandas as pd
import torch.utils.model_zoo as model_zoo

import os.path as pt
from models.segnet import segnet
from models.unet_model import UNet
from models.fusenet_model import FuseNet
from models.pspnet.psp_net import pspnet_10m

from utils.dataloader import train_xbd_data_loader
from utils.dataloader import val_xbd_data_loader


from models.damage.damage_net_vhr import damage_net_vhr
from models.damage.damaged_net_fusion_simple import damage_net_vhr_fusion_simple

from utils import resume

from torch.autograd import Variable
import cv2

RESULTS_PATH = os.environ["RESULTS_PATH"]  # "/results"
TESTDATA_PATH = os.environ["TESTDATA_PATH"] 

def main(
        batch_size,
        num_mini_batches,
        nworkers,
        datadir,
        outdir,
        num_epochs,
        snapshot,
        finetune,
        lr,
        n_classes,
        loadvgg,
        network_type,
        fusion,
        data,
        write,
        num_test
):

    n_classes = 2
    tile_size = 1024
    channel_basis = {'vhr': 3}
    channel_dict = dict()
    np.random.seed(0)
    network_type = 'vhr'

    for item in data:
        channel_dict['{}'.format(item)] = channel_basis[item]

    if network_type == 'vhr':
	network = pspnet_10m()
    if network_type == 'baseline_vhr':
        network = damage_net_vhr(n_classes=n_classes)
        network.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    elif network_type == 'baseline_s1':
        network = damage_net_s1(n_classes=n_classes)
        network.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    elif network_type == 'baseline_s2':
        network = damage_net_s2(n_classes=n_classes)
        network.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    elif network_type == 'damagenet_fusion_simple':
        network = damage_net_vhr_fusion_simple(n_classes=n_classes)
        network.load_state_dict(model_zoo.load_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth'))

    elif not finetune:
        finetune = RESULTS_PATH + "/vhr_buildings10m" + "/epoch_{:02}_classes_{:02}.pth".format(
            num_epochs, n_classes)
	
    if datadir is None:
        datadir = TESTDATA_PATH

    val = val_xbd_data_loader(datadir, batch_size=batch_size, num_workers=nworkers, mode='test')

    metric = classmetric.ClassMetric()

    if torch.cuda.is_available():
        network = network.cuda()

    if loadvgg == True:
        network.load_vgg16_weights()

    if torch.cuda.is_available():
       network = nn.DataParallel(network).cuda()
    # else:
    #    network = nn.DataParallel(network)

    param_groups = [
        {'params': network.parameters(), 'lr': lr}
    ]

    if finetune or snapshot:
        state = resume(finetune or snapshot, network, None)
    loss_str_list = []
    
    metric_dicts = []
    network.eval()
    for iteration, data in enumerate(val):	
        tile, input, target_tensor = data
        target = tensor_to_variable(target_tensor[0])
        
        for key in input.keys():
            input[key] = tensor_to_variable(input[key])
        
        output_raw = network.forward(input['vhr']) 
        
        if type(output_raw) == tuple:
            output_raw = output_raw[-1] 
            
        # force the output label map to match the target dimensions
        b, h, w = target.shape
        output_raw = torch.nn.functional.upsample(output_raw, size=(h, w), mode='bilinear')

        # Normalize
        if n_classes == 1:
            output = output_raw
        else:
            soft = nn.Softmax2d()
            output = soft(output_raw)
            
        train_metric = metric(target, output)
        
        if not write:
            metric_dicts.append(train_metric)
            if iteration > num_test - 1:
                break        
            continue

        loss_str_list.append("Input ID: {}; Metric: {} ".format(tile, str(train_metric)))

        # convert zo W x H x C
        if torch.cuda.is_available():
            prediction = output.cpu().data[0] #.permute(1, 2, 0)
            target = target.cpu().data[0] #.permute(1, 2, 0)
        else:
            prediction = output.cpu().data[0] #.permute(1, 2, 0)
            target = target.cpu().data[0] #.permute(1, 2, 0)

        if not os.path.exists(RESULTS_PATH+"/img"):
            os.makedirs(RESULTS_PATH+"/img")

        # Remove extra dim
        if n_classes == 1:
            prediction_img = prediction.numpy()
        else:
            prediction_img = np.argmax(prediction, prediction.shape.index(n_classes)).numpy()

        target_img = target.numpy()
        target_img[target_img == 1] = 2
        target_img[target_img == 0] = 1
        # Write input image (only first 3 bands)
        #input_img = input['vhr'].squeeze(0).cpu().numpy()
        
        #if input_img[:, 0, 0].size >= 3:
        #    input_img = cv2.merge((input_img[0], input_img[1], input_img[2]))
        #else:
        #   input_img = input_img[0]

        upsample = nn.Upsample(size=(int(tile_size/1.25), int(tile_size/1.25)), mode='bilinear', align_corners=True)  # Harvey
        
        #cv2.imwrite(RESULTS_PATH+"/img/{}_input_class_{:02}.png".format(iteration, n_classes), input['vhr'].cpu().data[0].numpy())
        cv2.imwrite(RESULTS_PATH+"/results_damage/test_damage_{}_prediction.png".format(tile[0].split('/')[-1].split('post_')[-1].split('.png')[0]), prediction_img)
    #    cv2.imwrite(RESULTS_PATH+"/img/{}_prediction_class_{:02}_{}.png".format(iteration, n_classes, tile[0].split('/')[-1]), prediction_img*255)
    #    cv2.imwrite(RESULTS_PATH+"/img/{}_target_class_{:02}_{}.png".format(iteration, n_classes, tile[0].split('/')[-1]), target_img*255)
        #exit(0)

        if not write:
            metrics_df = pd.DataFrame(metric_dicts)
            mean_metrics = df.mean()
            return dict(mean_metrics)

        with open(RESULTS_PATH + "/MSEloss.csv", "w") as output:
            writer = csv.writer(output, delimiter=';', lineterminator='\n')
            for val in loss_str_list:
                writer.writerow([val])
    if not write:
        metrics_df = pd.DataFrame(metric_dicts)
        mean_metrics = metrics_df.mean()
        return dict(mean_metrics)
 

def tensor_to_variable(tensor):
    if torch.cuda.is_available():
        return Variable(tensor).cuda()
    else:
        return Variable(tensor)

if __name__ == '__main__':
    import argparse

    text_type=str

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-b', '--batch-size',
        default=8,
        type=int,
        help='number of images in batch',
    )
    parser.add_argument(
        '-w', '--workers',
        default=1,
        type=int,
        help='number of workers',
    )
    parser.add_argument(
        '-o', '--outdir',
        default='.',
        type=text_type,
        help='output directory',
    )
    parser.add_argument(
        '-e', '--num-epochs',
        default=10,
        type=int,
        help='number of epochs',
    )
    parser.add_argument(
        '-m', '--num-mini-batches',
        default=1,
        type=int,
        help='number of mini batches per batch',
    )
    parser.add_argument(
        '-p', '--datadir',
        default='.',
        type=text_type,
        help='ILSVRC12 dir',
    )
    parser.add_argument(
        '-r', '--resume',
        type=text_type,
        help='snapshot path',
    )
    parser.add_argument(
        '-f', '--finetune',
        type=text_type,
        help='finetune path',
    )
    parser.add_argument(
        '--lr',
        default=0.001,
        type=float,
        help='initial learning rate',
    )
    parser.add_argument(
        '-c', '--n_classes',
        default = 1,
        type = int,
        help = 'prediction type: =1: regression />1: classification',
    )
    parser.add_argument(
        '-x', '--experiment',
        default="PspBuildingsRGB",
        type=text_type,
        help='experiment name',
    )
    parser.add_argument(
        '-v', '--loadvgg',
        default=0,
        type=int,
        help='load vgg16',
    )
    parser.add_argument(
        '-n', '--network_type',
        default='segnet',
        type=text_type,
        help='network type',
    )
    parser.add_argument(
        '-en', '--fusion',
        default='vhr',
        type=text_type,
        help='fusion type',
    )
    parser.add_argument(
        '-d', '--data',
        default=['vhr'],
        type=text_type,
        nargs='+',
        help='datasets used',
    )
    parser.add_argument(
	'-wr', '--write',
        default=True,
        type=bool,
        help='must be set to False except when used in best_epochs.py' 
    )
    parser.add_argument(
	'-nt', '--num_test',
	default=30,
	type=int,
	help='number of test examples to consider'
    )

    args, unknown = parser.parse_known_args()
    try:
        main(
            args.batch_size,
            args.num_mini_batches,
            args.workers,
            args.datadir,
            args.outdir,
            args.num_epochs,
            args.resume,
            args.finetune,
            args.lr,
            args.n_classes,
            args.loadvgg,
            args.network_type,
            args.fusion,
            args.data,
            args.write,
	    args.num_test
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
