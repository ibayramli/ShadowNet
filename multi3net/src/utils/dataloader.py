import torch
from torch.utils import data
from .image import tiff_to_nd_array, random_augment, normalize_channels, apply_blur
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import pandas as pd

transform = transforms.Compose([])

class XBDImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="/n/tambe_lab/disaster_relief/xBD_data/data",
                 transform=None,
                 mode='train'):
	if mode not in ['test', 'train', 'val']: 
            raise ValueError('Mode must be one of \'test\', \'train\', or \'val\'. ')

	self.mode = mode
        self.root_dir = root_dir
        self.n_classes = 2
        self.tile_size = 1024 
        self.validation = mode == 'val' 
        self.transform = transform
        self.posts = [pn for pn in os.listdir(os.path.join(self.root_dir, self.mode)) if '_post' in pn]
        self.pres = [pn for pn in os.listdir(os.path.join(self.root_dir, self.mode)) if '_pre' in pn]
        self.labels = [pn for pn in os.listdir(os.path.join(self.root_dir, 'masks')) if '_post' in pn]
        self.augmentation = random_augment() if self.validation == False else random_augment(0)

        if len(self.posts) != len(self.pres):
            raise ValueError('Number of post-disaster and pre-disaster images don\'t match')

    def __len__(self):
        return len(self.posts)

    def reduce_channels(self, arr):
        '''

	Turns a NxMxK tensor into MxK one

	'''
	reduced = torch.max(arr, dim=0)[0]
        reduced[reduced > 0] = 1
        
        return reduced 

    def multiclass_generation(self, img):  # FIXME
        for idx in range(len(img)):
            img[idx] = pd.cut(img[idx].flatten(), bins=self.n_classes, labels=False)

    def read_target_file(self, label_img_path):
        label = tiff_to_nd_array(label_img_path).astype(float)
        label = label.clip(min=0)
        label = self.augmentation(label)
        label = label.squeeze()

        label = torch.from_numpy(label).long()
        return label

    def path_to_tensor(self, vhr_img_path):
        def resize(array, shape):
            array = np.moveaxis(array, 0, 2)
            array = cv2.resize(array, dsize=shape, interpolation=cv2.INTER_LINEAR)

            return np.moveaxis(array, 2, 0)
	
	tile_size = self.tile_size
        h_vhr, w_vhr = int(tile_size*2/1.25), int(tile_size*2/1.25)

        img_vhr = tiff_to_nd_array(vhr_img_path).astype(float)
        img_vhr = resize(img_vhr, shape=(int(h_vhr / 2), int(w_vhr / 2)))
        img_vhr = self.augmentation(img_vhr)
        img_vhr = normalize_channels(img_vhr, 255, 0)
        img_vhr = torch.from_numpy(img_vhr)

        return img_vhr

    def __getitem__(self, idx):
        tile_size = self.tile_size
        h_vhr, w_vhr = int(tile_size*2/1.25), int(tile_size*2/1.25)

        inputs = dict()	
        
        post_cur = self.posts[idx]
        pre_cur = post_cur.replace('post', 'pre')
        vhr_post_path = os.path.join(self.root_dir, self.mode, post_cur)
        vhr_pre_path = os.path.join(self.root_dir, self.mode, pre_cur)
        post = self.path_to_tensor(vhr_post_path)
        #pre = self.path_to_tensor(vhr_pre_path)
        #inputs['pre_vhr'] = pre	
        inputs['post_vhr'] = post

        if post_cur in self.labels:
            label = self.read_target_file(os.path.join(self.root_dir, 'masks', post_cur))
            label = self.reduce_channels(label)
        else: # if label is missing, the label has no polygons (is all 0s)
            label = self.read_target_file(os.path.join(self.root_dir, 'masks', 'black_img.png'))
            label = self.reduce_channels(label)
        tile = os.path.join(self.root_dir, 'masks', post_cur)

        return tile, inputs, (label, )

def train_xbd_data_loader(root_dir, batch_size, num_workers, shuffle=True, use_multi_sar=False,
                             mode='train', labelimage="buildings10m.tif"):
    dataset = XBDImageDataset(root_dir,
                           transform=transform,
                           mode=mode)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)

    return dataloader

def val_xbd_data_loader(root_dir, batch_size, num_workers, shuffle=True, use_multi_sar=False,
                              mode='train', labelimage="buildings10m.tif"):
    dataset = XBDImageDataset(root_dir,
                           transform=transform,
                           mode=mode)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataloader
