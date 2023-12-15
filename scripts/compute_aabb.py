'''
@author: nmarticorena

This scripts compute the aabb of the scene, the approach relies on min max
operation on the agregated point cloud
'''

import open3d as o3d
import numpy as np
import os
import json

from dataclasses import dataclass
import tyro

from nerf_tools.utils.depth import get_pointcloud

@dataclass
class Args:
    dataset: str = "nerf" # Dataset to use
    dataset_path: str = '/home/nmarticorena/Documents/tools/RLBench/'  # Path to the dataset 
    save: bool = True # Save the aabb to the dataset


args = tyro.cli(Args)
if args.dataset == 'nerf':
    from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset
    from nerf_tools.dataset.nerf_dataset import load_from_json
    if ".json" not in args.dataset_path:
        args.dataset_path = os.path.join(args.dataset_path, 'transforms.json')
    with open(args.dataset_path, "r") as f:
        config = json.load(f)
    oDataset = load_from_json(args.dataset_path)
    camera_index = range(0, len(oDataset.frames))
    oDataset.set_frames_index(camera_index)
else:
    from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset as Dataset

pcd = get_pointcloud(oDataset, 2.0, 1, 100)

aabb = pcd.get_axis_aligned_bounding_box()

min_bound = aabb.min_bound - np.array([0.5, 0.5, 0])
max_bound = aabb.max_bound + np.array([0.5, 0.5, 0])

aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

aabb.color = (1,0,0)


o3d.visualization.draw_geometries([pcd, aabb])

aabb_array = np.array([aabb.min_bound - 0.5, aabb.max_bound + 0.5])
config["aabb"] = aabb_array.tolist()

print(config)
json_object = json.dumps(config, indent = 4)

if args.save:
    with open(args.dataset_path, 'w') as f:
        json.dump(config, f, indent = 4)
    
