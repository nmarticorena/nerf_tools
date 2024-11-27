"""
@author: nmarticorena

This scripts compute the aabb of the scene, the approach relies on min max
operation on the agregated point cloud
"""

import open3d as o3d
import numpy as np
import os
import json
import tyro

from nerf_tools.utils.depth import get_pointcloud
import nerf_tools.configs as configs
from nerf_tools.utils.colmap_tools import (
    point_cloud_saver,
    camera_poses_saver,
    camera_info_saver,
)

args = tyro.cli(configs.AABB)


if args.dataset.type == "nerf":
    from nerf_tools.dataset.nerf_dataset import load_from_json

    dataset_path = args.dataset.dataset_path

    if ".json" not in dataset_path:
        dataset_path = os.path.join(dataset_path, "transforms.json")
    with open(dataset_path, "r") as f:
        config = json.load(f)
    oDataset = load_from_json(dataset_path)
    camera_index = range(0, len(oDataset.frames))
    oDataset.set_frames_index(camera_index)

pcd = get_pointcloud(
    oDataset,
    max_depth=args.pcd.max_depth,
    skip_frames=args.pcd.skip_frames,
    filter_step=args.pcd.down_sample_frames,
    voxel_size=args.pcd.down_sample_voxel_size,
)


aabb = pcd.get_axis_aligned_bounding_box()

min_bound = aabb.min_bound - np.array([args.extra, args.extra, 0])
max_bound = aabb.max_bound + np.array([args.extra, args.extra, 0])

aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

aabb.color = (1, 0, 0)

if args.gui:
    cameras = oDataset.draw_cameras()
    final = [pcd, aabb, *cameras]
    o3d.visualization.draw_geometries(final)

aabb_array = np.array([aabb.min_bound - args.extra,
                      aabb.max_bound + args.extra])
config["aabb"] = aabb_array.tolist()

json_object = json.dumps(config, indent=4)
# R = sm.SE3.Rx(np.pi/2) * sm.SE3.Ry(np.pi/2)
# pcd = pcd.transform(R.A)

o3d.io.write_point_cloud("test.ply", pcd)

os.makedirs(os.path.join(args.dataset.dataset_path,
            "sparse", "0"), exist_ok=True)

point_cloud_saver(pcd, os.path.join(args.dataset.dataset_path, "sparse/0/"))
camera_poses_saver(oDataset, os.path.join(
    args.dataset.dataset_path, "sparse/0/"))
camera_info_saver(oDataset, os.path.join(
    args.dataset.dataset_path, "sparse/0/"))

if args.save:
    with open(dataset_path, "w") as f:
        json.dump(config, f, indent=4)
