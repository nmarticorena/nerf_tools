"""
@author: nmarticorena

This scripts plot the dataset as a depth point cloud in open3D to check the
dataset is correct
"""

import open3d as o3d
import os
import json
from nerf_tools.utils.depth import get_tsdf

from dataclasses import dataclass
import tyro
from nerf_tools.configs import DatasetPath


args = tyro.cli(DatasetPath)


if args.type == "nerf":
    from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset
    from nerf_tools.dataset.nerf_dataset import load_from_json

    oDataset = load_from_json(args.dataset_path)
else:
    from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset as Dataset

cameras = oDataset.draw_cameras()

o3d.visualization.draw_geometries([get_tsdf(oDataset), *cameras])
