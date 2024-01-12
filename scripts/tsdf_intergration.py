'''
@author: nmarticorena

This scripts plot the dataset as a depth point cloud in open3D to check the 
dataset is correct
'''

import open3d as o3d
import os
import json
from nerf_tools.utils.depth import get_tsdf

from dataclasses import dataclass
import tyro


@dataclass
class Args:
    dataset: str = "nerf" # Dataset to use
    dataset_path: str = '/home/nmarticorena/Documents/tools/RLBench/'  # Path to the dataset 
args = tyro.cli(Args)


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

cameras = oDataset.draw_cameras()

o3d.visualization.draw_geometries([get_tsdf(oDataset), *cameras])



