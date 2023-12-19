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
from nerf_tools.configs import *

args = tyro.cli(AABB)


if args.dataset.type == 'nerf':
    from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset
    from nerf_tools.dataset.nerf_dataset import load_from_json
    
    dataset_path = args.dataset.dataset_path
    
    if ".json" not in dataset_path:
        dataset_path = os.path.join(dataset_path, 'transforms.json')
    with open(dataset_path, "r") as f:
        config = json.load(f)
    oDataset = load_from_json(dataset_path)
    camera_index = range(0, len(oDataset.frames))
    oDataset.set_frames_index(camera_index)
else:
    from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset as Dataset

pcd = get_pointcloud(oDataset, 2.0, args.pcd.skip_frames, 
                     args.pcd.down_sample_frames,
                     voxel_size= args.pcd.down_sample_voxel_size)


aabb = pcd.get_axis_aligned_bounding_box()

min_bound = aabb.min_bound - np.array([args.extra, args.extra, 0])
max_bound = aabb.max_bound + np.array([args.extra, args.extra, 0])

aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

aabb.color = (1,0,0)


o3d.visualization.draw_geometries([pcd, aabb])

aabb_array = np.array([aabb.min_bound - args.extra, aabb.max_bound + args.extra])
config["aabb"] = aabb_array.tolist()

print(config)
json_object = json.dumps(config, indent = 4)

if args.save:
    with open(dataset_path, 'w') as f:
        json.dump(config, f, indent = 4)
    
