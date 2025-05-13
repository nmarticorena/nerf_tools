"""
@author: nmarticorena

This scripts plot the dataset as a depth point cloud in open3D to check the
dataset is correct
"""

import open3d as o3d
from nerf_tools.utils.depth import get_tsdf
import numpy as np

import tyro
from nerf_tools.configs import TSDF
import trimesh

args = tyro.cli(TSDF)


if args.dataset.type == "nerf":
    from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset
    from nerf_tools.dataset.nerf_dataset import load_from_json

    oDataset = load_from_json(args.dataset.dataset_path)
else:
    from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset as Dataset

    exit("Need to fix")

cameras = oDataset.draw_cameras()

mesh = get_tsdf(oDataset, depth_trunc=args.depth_trunc, voxel_size=args.voxel_size)

if args.save:
    path = f"results/meshes/{args.name}.ply"
    print(f"saving on {path}")
    o3d.io.write_triangle_mesh(path, mesh)

if args.gui:
    o3d.visualization.draw_geometries([mesh, *cameras])
