"""
@file: scripts/isdf_vs_pcd.py

This script is an experiment to compare the iSDF and the PCD methods.

"""

from nerf_tools.configs import *
from nerf_tools.utils.utils import load_pcd, load_dataset
from nerf_tools.tools import *
import tyro
import os
import trimesh

@dataclass
class Args:
    dataset: DatasetPath = DatasetPath()
    pcd: SamplePCD = SamplePCD()
    results: str = "results"




args = tyro.cli(Args)
os.makedirs(args.results, exist_ok=True)

dataset = load_dataset(args.dataset)
# pcd = load_pcd(args.dataset, args.pcd)

# np.save("pcd.npy", np.asarray(pcd.points))

pcd = np.load("pcd.npy")

model = min_sdf(pcd)

aabb = np.array(dataset.aabb)

grid,scale = grid_from_aabb(aabb, 250)

levels = [0.03, 0.05,0.1]

meshes = marching_cubes(grid, model, scale, 250, aabb, levels = levels)

import open3d as o3d
for ix,mesh in enumerate(meshes):
    o3d.visualization.draw_geometries([mesh.as_open3d]) 
    trimesh.exchange.export.export_mesh(mesh, os.path.join(args.results, f"mesh_{levels[ix]}.ply"))


