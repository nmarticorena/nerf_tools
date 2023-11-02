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
from nerf_tools.utils.depth import get_pointcloud

parser = argparse.ArgumentParser(description='Check dataset')
parser.add_argument('--dataset', type=str, default='nerf', help='dataset to use')
parser.add_argument('--dataset_path', type=str, 
                      default='/home/nmarticorena/Documents/tools/RLBench/', 
                      help='path to the dataset')

args = parser.parse_args()

if args.dataset == 'nerf':
    from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset
    from nerf_tools.dataset.nerf_dataset import load_from_json
    if ".json" not in args.dataset_path:
        args.dataset_path = os.path.join(args.dataset_path, 'transforms.json')

    oDataset = load_from_json(args.dataset_path) 
    camera_index = range(0, len(oDataset.frames))
    oDataset.set_frames_index(camera_index)
else:
    from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset as Dataset


o3d.visualization.draw_geometries([get_pointcloud(oDataset, skip_frames= 10)])



