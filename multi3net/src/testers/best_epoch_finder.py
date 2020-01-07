import os
import argparse
import numpy as np

from predict import main as get_epoch_metrics

NUM_CLASSES = 2

def get_best_weights(results_path='', weights=[2, 3, 4, 5, 6, 8, 9, 10, 'theoretical']):
    if not results_path:
        results_path = os.environ["RESULTS_PATH"]
    weight_dict = {}
    for w in weights:
        print('Computing best epoch for weight {}'.format(w))
        if w != 'theoretical':
	    path = os.path.join(results_path, 'predictions_1_' + str(w))
        else:
	    path = os.path.join(results_path, 'predictions_' + w)
	max_epoch, best_score = get_best_epoch_metrics(path)
        weight_dict[w] = (max_epoch, best_score)
        print(weight_dict)
   
    print(weight_dict) 
    return weight_dict	

def get_best_epoch_metrics(predictions_path=""):
    if predictions_path:
 	predictions_path = os.environ["RESULTS_PATH"]
    
    path = os.path.join(predictions_path, 'vhr_buildings10m')
    subdirs = [os.path.join(path, name) for name in os.listdir(path) if name.endswith('.pth')]
    subdirs.sort()
    
    epoch_scores = {}   
    for i, sub in enumerate(subdirs):
        print('Computing test output of epoch {}'.format(str(i)))
	epoch_metrics = get_epoch_metrics(
				batch_size=8,
				num_mini_batches=1,
			        nworkers=1,
			        datadir=None,
			        outdir='.',
			        num_epochs=10,
			        snapshot=None,
			        finetune=sub,
	        		lr=0.001,
		        	n_classes=2,
			        loadvgg=0,
			        network_type='segnet',
			        fusion='vhr',
			        data=['vhr'],
			        write=False,
			        num_test=20  
			)

	score = epoch_metrics['f1_building'] + epoch_metrics['iou_building']
	epoch_scores[i] = score
   	print(epoch_metrics)
	print('The f_1 + iou_building score is: {}'.format(score))

    max_epoch = max(epoch_scores.iterkeys(), key=(lambda key: epoch_scores[key]))
    print('The best epoch is: {} with a epoch score of {}'.format(max_epoch, epoch_scores[max_epoch]))
    return (max_epoch, epoch_scores[max_epoch])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-p', '--path',
        type=str,
        help='path to the results folder, if not given defaults to the environmental variable RESULTS_PATH',
    )
    args = parser.parse_args()

    try:
	get_best_epoch_metrics(args.path)
    except KeyboardInterrupt:
	pass
    finally:
	print()
