import os

import torch
import tyro
import open3d as o3d

from nerf_tools.configs import GSplatLoader
from nerf_tools.utils.math import quaternion_to_rotation_matrix
from nerf_tools.utils.meshes import create_gs_mesh

args = tyro.cli(GSplatLoader)

path = args.path + "/ckpts"

if args.list:
    print("the files " ,os.listdir(path) ," are available")
    exit(0)


for step in args.step:
    gsplat_path = os.path.join(path, f"ckpt_{step-1}.pt")


    try:
        splats = torch.load(gsplat_path, weights_only= True, map_location =torch.device("cpu"))["splats"]
    except FileNotFoundError:
        print(f"File {gsplat_path} not found.")
        exit(1)

    means:torch.Tensor = splats["means"]
    opacities = torch.nn.Sigmoid()(splats["opacities"])
    quat = torch.nn.functional.normalize(splats["quats"])
    scales = torch.exp(splats["scales"])
    if not args.is_3d:
        scales[:, -1] = 0. # Flatten the ellipsoid to a 2D plane
    sh0 = splats["sh0"]
    shn = splats["shN"]

    rots = quaternion_to_rotation_matrix(quat)

    print(f"Loading {means.shape[0]} splats with")
    mesh = create_gs_mesh(means.cpu().numpy(),
                        rots.cpu().numpy(),
                        scales.cpu().numpy(),
                        torch.zeros_like(means).cpu().numpy(),
                        res = args.res)

    print(rots)
    mesh.compute_vertex_normals()
    print(mesh)

    if args.visualize:
        o3d.visualization.draw_geometries([mesh])


    if args.save:
        save_name = os.path.join(args.save_path, f"mesh_{step}.ply")
        success = o3d.io.write_triangle_mesh(save_name, mesh, print_progress=True)

