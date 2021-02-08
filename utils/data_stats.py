from __future__ import division
import os
import sys

DISASTER_TYPES = ['volcano', 'hurricane', 'fire', 'tsunami', 'earthquake', 'monsoon', 'flood', 'tornado']
NUM_DISASTER_EVENTS = 19

def get_disaster_names(path='../data/train'):
    data = os.listdir(path)
    dis_names = set([d.split('_')[0] for d in data])
  
    assert len(dis_names) == NUM_DISASTER_EVENTS

    return dis_names


def get_disaster_count(path):
    dis_names = get_disaster_names()
    dis_dict = dict()
    for name in dis_names:
	dis_dict[name] = 0	

    data = os.listdir(path)
    for d in data:
	dis = d.split('_')[0]
	dis_dict[dis] += 1

    return dis_dict


def get_disaster_ratio(path):
    dis_count = get_disaster_count(path)
    path_total = len(os.listdir(path))

    dict_total = 0
    for key in dis_count.keys():
	d = dis_count[key]
	dict_total += d
	dis_count[key] = round(d/path_total, 2)

    assert dict_total == path_total

    return dis_count


def get_disaster_type_count(path):
    dis_count = get_disaster_count(path)

    dis_type = dict()
    for dis in DISASTER_TYPES:
	for key in dis_count.keys():
	    if dis in key:
		if dis in dis_type.keys():
		    dis_type[dis] += dis_count[key] 
		else:
		    dis_type[dis] = dis_count[key]

    return dis_type

def get_disaster_type_ratio(path):
    dis_count = get_disaster_type_count(path)
    path_total = len(os.listdir(path))

    dict_total = 0
    for key in dis_count.keys():
        d = dis_count[key]
        dict_total += d
        dis_count[key] = round(d/path_total, 2)

    assert dict_total == path_total

    return dis_count


def print_data_stats(paths):
    for path in paths:
	print('Disaster counts for' + path  + ' is:', get_disaster_count(path), '\n')
        print('Disaster ratio for' + path  + ' is:', get_disaster_ratio(path), '\n')
        print('Disaster counts by type for' + path  + ' is:', get_disaster_type_count(path), '\n')
        print('Disaster ratio by type for' + path  + ' is:', get_disaster_type_ratio(path), '\n')

if __name__ == '__main__':
    paths = ['../data/val', '../data/train', '../data/test']
    
    sys.stdout = open('output.txt', 'w')
    try:
        print_data_stats(paths)
    except KeyboardInterrupt:
	pass
