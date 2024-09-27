"""
@author: nmarticorena

Show the cameras of the recorded dataset
"""

import pdb
import open3d as o3d
import numpy as np
import os
import json

from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset
from dataclasses import dataclass
import tyro
import time


# set printing
np.set_printoptions(precision=4, suppress=True)


@dataclass
class Args:
    dataset: str = "nerf"  # Dataset to use
    # dataset_path: str = '/media/nmarticorena/DATA/datasets/NeRFCapture/cupboard2/' # Path to the dataset
    dataset_path: str = "/home/nmarticorena/Documents/papers/mobile_manipulation_neo/results/real_scans/"
    # dataset_path: str = '/media/nmarticorena/DATA/nerf_standard/bookshelf_ensemble'
    env_name: str = "s12"  # Name of the environment


isdf_path: str = "/home/nmarticorena/Documents/papers/neo-sddf/neo_sddf/dependencies/iSDF/results/iSDF/{}/0/meshes/last.stl"


args = tyro.cli(Args)

mesh = o3d.io.read_triangle_mesh(isdf_path.format(args.env_name))
mesh.compute_vertex_normals()

with open(os.path.join(args.dataset_path, args.env_name, "transforms.json")) as f:
    config = json.load(f)

config["folder"] = args.dataset_path
# import pdb; pdb.set_trace()

oDataset = Dataset(**config)


vis = o3d.visualization.Visualizer()
vis.create_window(
    width=oDataset.w, height=oDataset.h, visible=True, window_name="NeRF Dataset"
)

view_control = vis.get_view_control()


# vis.pool_events()
vis.update_renderer()
camera_parameters = view_control.convert_to_pinhole_camera_parameters()

vis.add_geometry(mesh)

vis.poll_events()
vis.update_renderer()


# Capture the rendered image

i = 0

while True:
    params = view_control.convert_to_pinhole_camera_parameters()

    i = (i + 1) % len(oDataset.frames)
    # i = (i + 1)

    camera_parameters.extrinsic = oDataset.get_transforms_cv2([i])[0]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(oDataset.w),
        int(oDataset.h),
        oDataset.fl_x,
        oDataset.fl_y,
        oDataset.cx,
        oDataset.cy,
    )
    intrinsic.set_intrinsics(
        int(oDataset.w),
        int(oDataset.h),
        oDataset.fl_x,
        oDataset.fl_y,
        oDataset.cx,
        oDataset.cy,
    )
    camera_params = o3d.camera.PinholeCameraParameters()
    # camera_params.intrinsic = intrinsic
    camera_params.extrinsic = oDataset.get_transforms_cv2([i])[0]
    print(camera_params.extrinsic)
    pdb.set_trace()
    # print(intrinsic.intrinsic_matrix)
    view_control.convert_from_pinhole_camera_parameters(camera_params, True)
    vis.update_renderer()
    vis.poll_events()
    # view_control.scale(0.1)
    # vis.update_renderer()
    # time.sleep(1)

    # image = vis.capture_screen_float_buffer()
    #
    # img_np = np.asarray(image)
    #
    # # Display the image (you can save it or process it further as needed)
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(img_np)
    # plt.axis("off")
    # plt.savefig(f"image{i}.png")
    # plt.imshow(img_np)
    # plt.show()
