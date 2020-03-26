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
                 mode='train',
                 experiment='post'):

	if mode not in ['test', 'train', 'val']: 
            raise ValueError('Mode must be one of \'test\', \'train\', or \'val\'. ')
	
	self.experiment = experiment
	self.mode = mode
        self.root_dir = root_dir
        self.n_classes = 2
        self.tile_size = 1024 
        self.validation = mode != 'train' 
        self.transform = transform
        self.augmentation = random_augment() if self.validation == False else random_augment(0)
        self.posts = [pn for pn in os.listdir(os.path.join(self.root_dir, self.mode)) if '_post' in pn]
        self.pres = [pn for pn in os.listdir(os.path.join(self.root_dir, self.mode)) if '_pre' in pn]
        self.labels = [pn for pn in os.listdir(os.path.join(self.root_dir, 'masks'))]
        

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

    def array_to_tensor(self, img_vhr):
        def resize(array, shape):
            array = np.moveaxis(array, 0, 2)
            array = cv2.resize(array, dsize=shape, interpolation=cv2.INTER_LINEAR)

            return np.moveaxis(array, 2, 0)

        tile_size = self.tile_size
        h_vhr, w_vhr = int(tile_size*2/1.25), int(tile_size*2/1.25)

        img_vhr = resize(img_vhr, shape=(int(h_vhr / 2), int(w_vhr / 2)))
        img_vhr = self.augmentation(img_vhr)
        img_vhr = normalize_channels(img_vhr, 255, 0)
        img_vhr = torch.from_numpy(img_vhr)

        return img_vhr

    def get_pre_prime(self, pre, post, mask):
        pre_prime = pre
        if torch.cuda.is_available():
            mask = mask.cpu().data.numpy()
	    mask = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
        else:
            mask = mask.data.numpy()
	    mask = np.repeat(mask[np.newaxis, :, :], 3, axis=0)
        pre_prime[mask == 0] = post[mask == 0]
	
        return self.array_to_tensor(pre_prime)

    def __getitem__(self, idx):
        tile_size = self.tile_size
        h_vhr, w_vhr = int(tile_size*2/1.25), int(tile_size*2/1.25)

        inputs = dict()	

        post_cur = self.posts[idx]
        vhr_post_path = os.path.join(self.root_dir, self.mode, post_cur)
        post = tiff_to_nd_array(vhr_post_path).astype(float)

        pre_cur = post_cur.replace('post', 'pre')
        vhr_pre_path = os.path.join(self.root_dir, self.mode, pre_cur)
        pre = tiff_to_nd_array(vhr_pre_path).astype(float)

        match = pre_cur if self.experiment == 'pre' else post_cur
        if match in self.labels:
            label = self.read_target_file(os.path.join(self.root_dir, 'masks', match))
            label = self.reduce_channels(label)
        else: # if label is missing, the label has no polygons (is all 0s)
            label = self.read_target_file(os.path.join(self.root_dir, 'masks', 'black_img.png'))
            label = self.reduce_channels(label)

        if self.experiment == 'post':
            inputs['vhr'] = self.array_to_tensor(post)
        elif self.experiment == 'pre':
            inputs['vhr'] = self.array_to_tensor(pre)
        elif self.experiment == 'pre_post':
            inputs['vhr_pre'] = self.array_to_tensor(pre)
            inputs['vhr_post'] = self.array_to_tensor(post)
        elif self.experiment == 'pre_post_experimental':
            inputs['vhr_pre'] = self.get_pre_prime(pre, post, label)
            inputs['vhr_post'] = self.array_to_tensor(post)
        elif self.experiment == 'prime_transform':
            inputs['vhr_pre'] = self.array_to_tensor(pre)
            inputs['vhr_post'] = self.array_to_tensor(post)
            label = self.get_pre_prime(pre, post, label)
            tile = os.path.join(self.root_dir, 'masks', match)

            return tile, inputs, (label, )

        tile = os.path.join(self.root_dir, 'masks', match)

        return tile, inputs, (label, )

def train_xbd_data_loader(root_dir, batch_size, num_workers, shuffle=True, use_multi_sar=False,
                             mode='train', labelimage="buildings10m.tif", experiment='post'):
    dataset = XBDImageDataset(root_dir,
                           transform=transform,
                           mode=mode, 
                           experiment=experiment)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)

    return dataloader

def val_xbd_data_loader(root_dir, batch_size, num_workers, shuffle=True, use_multi_sar=False,
                              mode='train', labelimage="buildings10m.tif", experiment='post'):
    dataset = XBDImageDataset(root_dir,
                           transform=transform,
                           mode=mode, 
                           experiment=experiment)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataloader

