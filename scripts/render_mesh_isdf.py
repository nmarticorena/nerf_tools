'''
@author: nmarticorena

Show the cameras of the recorded dataset
'''
import pdb
import open3d as o3d
import numpy as np
import os
import json

from dataclasses import dataclass
import tyro

@dataclass
class Args:
    dataset: str = "nerf" # Dataset to use
    # dataset_path: str = '/media/nmarticorena/DATA/datasets/NeRFCapture/cupboard2/' # Path to the dataset 
    dataset_path: str = '/media/nmarticorena/DATA/nerf_standard/bookshelf_ensemble'
    isdf_path: str = "/home/nmarticorena/Documents/papers/neo-sddf/neo_sddf/dependencies/iSDF/results/iSDF/bookshelf/0/meshes/last.stl"

args = tyro.cli(Args)

mesh = o3d.io.read_triangle_mesh(args.isdf_path)
mesh.compute_vertex_normals()
from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset

with open(os.path.join(args.dataset_path, 'transforms.json')) as f:
    config = json.load(f)

config['folder'] = args.dataset_path
# import pdb; pdb.set_trace()

oDataset = Dataset(**config)


vis = o3d.visualization.Visualizer()
vis.create_window(width = oDataset.w, height = oDataset.h, visible = True, window_name = 'NeRF Dataset' )

view_control = vis.get_view_control()
import pdb; pdb.set_trace()


# vis.pool_events()
vis.update_renderer()
camera_parameters = view_control.convert_to_pinhole_camera_parameters()

camera_parameters = o3d.camera.PinholeCameraParameters() 
# camera_parameters.intrinsic.set_intrinsics( 800, 600, oDataset.fl_x, oDataset.fl_y, oDataset.cx, oDataset.cy)
camera_parameters.intrinsic = oDataset.get_camera()

# camera_parameters = o3d.camera.PinholeCameraParameters()
# pdb.set_trace()
# camera_parameters.intrinsic = oDataset.get_camera()

camera_parameters.extrinsic = oDataset.get_transforms_cv2([4])[0]
print(camera_parameters.extrinsic)

view_control.convert_from_pinhole_camera_parameters(camera_parameters, True)

vis.add_geometry(mesh)

# vis.update_geometry()
vis.poll_events()
vis.update_renderer()


# Capture the rendered image
image = vis.capture_screen_float_buffer()

i = 0

while True:
    i = (i +1) % len(oDataset.frames)
    # i = (i + 1)

    camera_parameters.extrinsic = oDataset.get_transforms_cv2([i])[0]
    view_control.convert_from_pinhole_camera_parameters(camera_parameters, True)
    view_control.scale(0.1)
    pdb.set_trace()
    vis.poll_events()
    vis.update_renderer()

# Close the visualizer
# vis.destroy_window()

# Convert the captured image to a numpy array
import numpy as np
img_np = np.asarray(image)

# Display the image (you can save it or process it further as needed)
import matplotlib.pyplot as plt
plt.imshow(img_np)
plt.axis('off')
plt.show()






