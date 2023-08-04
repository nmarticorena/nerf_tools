'''
@author: nmarticorena

This scripts plot the dataset as a depth point cloud in open3D to check the 
dataset is correct
'''

import open3d as o3d
import numpy as np
import os
import json
import argparse
from nerf_tools.utils.depth import get_tsdf


parser = argparse.ArgumentParser(description='Check dataset')
parser.add_argument('--dataset', type=str, default='nerf', help='dataset to use')
parser.add_argument('--dataset_path', type=str, 
                      default='/home/nmarticorena/Documents/tools/RLBench/', 
                      help='path to the dataset')

args = parser.parse_args()

if args.dataset == 'nerf':
    from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset

    with open(os.path.join(args.dataset_path, 'transforms.json')) as f:
        config = json.load(f)

    config['folder'] = args.dataset_path
    oDataset = Dataset(**config)
    camera_index = range(0, len(oDataset.frames))
    oDataset.set_frames_index(camera_index)
else:
    from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset as Dataset


o3d.visualization.draw_geometries([get_tsdf(oDataset)])



