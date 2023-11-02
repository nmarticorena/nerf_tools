'''
@author: nmarticorena

Show the cameras of the recorded dataset
'''

import open3d as o3d
import numpy as np
import os
import json

from dataclasses import dataclass
import tyro

@dataclass
class Args:
    dataset: str = "nerf" # Dataset to use
    dataset_path: str = '/home/nmarticorena/Documents/tools/RLBench/'  # Path to the dataset 
args = tyro.cli(Args)

from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset

with open(os.path.join(args.dataset_path, 'transforms.json')) as f:
    config = json.load(f)

config['folder'] = args.dataset_path
oDataset = Dataset(**config)

Lines = oDataset.draw_cameras()
o3d.visualization.draw_geometries(Lines)