import os
import argparse
import numpy as np
from PIL import Image

def measure_imbalance(root_dir='/n/tambe_lab/disaster_relief/xBD_data/data/masks', sample_size=100, verbose=True, multi3net_imsize=0):
    if multi3net_imsize not in [0, 1, 2, 10]: raise ValueError('Multi3net only supports the following image sizes: 0, 1, 2, 10')
    img_dirs = np.array([pn for pn in os.listdir(root_dir)])
    aranged = np.arange(len(img_dirs))
    np.random.shuffle(aranged)
    
    img_dirs_sample = img_dirs[aranged[ : sample_size]]
    imbalances = np.zeros((len(img_dirs_sample), 1))
    for i, imdir in enumerate(img_dirs_sample):
        if verbose: print('Calculating imbalance for image {}.'.format(i + 1))
	path = os.path.join(root_dir, imdir) if not multi3net_imsize else os.path.join(root_dir, imdir, 'buildings' + str(multi3net_imsize) + 'm.tif')
	img = Image.open(path)
	img_arr = np.array(img)
        img_arr[img_arr > 0] = 1
	positive = np.sum(img_arr)
        total = img_arr.size
       
	imbalances[i,0] = positive/total
	
    print(imbalances.mean())
    return imbalances.mean()          

def get_imbalance_dist(root_dir='/n/tambe_lab/disaster_relief/xBD_data/data/masks', num_samples=20, sample_size=100, verbose=True, multi3net_imsize=0):
    imbalance_dist = np.zeros((1, num_samples))
    for i in range(num_samples):
        if verbose: print('Iteration ', i, ' ...')
	cur_imbalance = measure_imbalance(root_dir, sample_size, verbose, multi3net_imsize)
        imbalance_dist[0, i] = cur_imbalance
    
    print(imbalance_dist)
    np.save('imbalance_dist', imbalance_dist)
    return imbalance_dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Measure the class inbalance in an image dataset")
    parser.add_argument('--root_dir', type=str, default='/n/tambe_lab/disaster_relief/xBD_data/data/masks', help="the location of ground truth label images")
    parser.add_argument('--sample_size', type=str, default=100, help="the size of the sample to measure imbalance based on") 
    parser.add_argument('--v', type=bool, default=True, help="program talks a lot")
    parser.add_argument('--multi3net_imsize', type=int, default=0, help="size of the multi3net label image to compute imbalance based on") 
    parser.add_argument('--n', type=int, default=0, help="the number of random label samples to consider")    
 
    args = parser.parse_args()
    
    try:
        if args.n > 0:
            get_imbalance_dist(root_dir=args.root_dir, num_samples=args.n, sample_size=args.sample_size, verbose=args.v, multi3net_imsize=args.multi3net_imsize)
        else:
            measure_imbalance(args.root_dir, args.sample_size, args.v, args.multi3net_imsize)
    except KeyboardInterrupt:
        pass
    

