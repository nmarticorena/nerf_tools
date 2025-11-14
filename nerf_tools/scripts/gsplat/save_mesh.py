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
    C0 = 0.28209479177387814

    mask = opacities > args.min_opacity
    means = means[mask]
    quat = quat[mask]
    scales = scales[mask]
    sh0 = sh0[mask]
    shn = shn[mask]

    def SH2RGB(sh):
        return sh * C0 + 0.5

    rgb = SH2RGB(sh0)
    rgb = rgb.clamp(0, 1)
    # rgb = sh0

    if args.id_color:
        # Use the id color for the splats
        rgb = torch.rand(rgb.shape, device=rgb.device).unsqueeze(1)
        rgb = rgb.float()

    rots = quaternion_to_rotation_matrix(quat)

    print(f"Loading {means.shape[0]} splats with")
    mesh = create_gs_mesh(means.cpu().numpy(),
                        rots.cpu().numpy(),
                        scales.cpu().numpy(),
                        colors = rgb.squeeze().cpu().numpy(),
                        res = args.res)

    mesh.compute_vertex_normals()

    if args.visualize:
        o3d.visualization.draw_geometries([mesh])


    if args.save:
        save_name = os.path.join("results/ellipsoids",args.save_path, f"mesh_{step}.ply")
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        success = o3d.io.write_triangle_mesh(save_name, mesh, print_progress=False)

