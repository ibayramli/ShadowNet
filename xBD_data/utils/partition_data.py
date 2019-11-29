import os
import numpy as np
import shutil

def split_data(ratios, root_dir='/n/tambe_lab/disaster_relief/xBD_data/train'):
    if type(ratios) != tuple or len(ratios) != 3: raise ValueError('Ratios must be a tuple object with 3 entries')
    for rat in ratios: 
        if type(rat) != int: 
            raise ValueError('Ratios must be integers')

    img_pres = np.array([pn for pn in os.listdir(root_dir + '/images') if '_pre' in pn])
    img_posts = np.array([pn for pn in os.listdir(root_dir + '/images') if '_post' in pn])
   
    img_pres.sort()
    img_posts.sort()
 
    print('Number of pre objects is {}'.format(len(img_pres)))
    print('Number of post objects is {}'.format(len(img_posts)))

    if len(img_pres) != len(img_posts): raise ValueError('Number of pre-disaster images and post-disaster images do not match')

    ratios_sum = sum(ratios)
    num_ims = len(img_pres)
    single_unit = np.floor(num_ims/ratios_sum)
    if num_ims < ratios_sum: raise ValueError('Sum or ratios needs to be smaller than number of training images.')

    aranged = np.arange(0, num_ims)
    np.random.shuffle(aranged)

    train_len = int(single_unit*ratios[0])
    val_len = int(single_unit*ratios[1])
    test_len = int(single_unit*ratios[2])

    train_idx = aranged[0:train_len]
    val_idx = aranged[train_len : train_len + val_len]
    test_idx = aranged[train_len + val_len : ]
    
    train = np.concatenate([img_pres[train_idx], img_posts[train_idx]])
    val = np.concatenate([img_pres[val_idx], img_posts[val_idx]])   
    test = np.concatenate([img_pres[test_idx], img_posts[test_idx]])
   
    train.sort()
    val.sort()
    test.sort()

    
    for t in train:
        print('Copying {} ...'.format(t))
        shutil.copy(os.path.join(root_dir, 'images', t), os.path.join(root_dir, 'train', t))
    for t in val:
        print('Copying {} ...'.format(t))
        shutil.copy(os.path.join(root_dir, 'images', t), os.path.join(root_dir, 'val', t))
    for t in test:
        print('Copying {} ...'.format(t)) 
        shutil.copy(os.path.join(root_dir, 'images', t), os.path.join(root_dir, 'test', t)) 
    
split_data((7, 1, 2))
