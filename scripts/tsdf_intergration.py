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

mesh = get_tsdf(oDataset, depth_trunc=args.depth_trunc)

if args.save:
    path = f"results/meshes/{args.name}.ply"
    print(f"saving on {path}")
    o3d.io.write_triangle_mesh(path, mesh)
    # vertices = np.asarray(mesh.vertices)
    # faces = np.asarray(mesh.triangles)
    # colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    #
    # trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    # trimesh_mesh.export(path)

if args.visualize:
    o3d.visualization.draw_geometries([mesh, *cameras])
