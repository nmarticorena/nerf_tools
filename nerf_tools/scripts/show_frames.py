"""
@author: nmarticorena

This script shows frame by frame of the dataset

todo:
- Add that the camera starts on the center of the scene

"""

import time
import open3d as o3d
import numpy as np
import os
import json
import spatialmath as sm
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading

from dataclasses import dataclass
import tyro

from nerf_tools.utils.depth import get_frame
from nerf_tools.configs import AABB

args = tyro.cli(AABB)


if args.dataset.type == "nerf":
    from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset
    from nerf_tools.dataset.nerf_dataset import load_from_json

    dataset_path = args.dataset.dataset_path

    if ".json" not in dataset_path:
        dataset_path = os.path.join(dataset_path, "transforms.json")
    with open(dataset_path, "r") as f:
        config = json.load(f)
    oDataset = load_from_json(dataset_path)
    camera_index = range(0, len(oDataset.frames))
    oDataset.set_frames_index(camera_index)
else:
    from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset as Dataset



class Open3DFrameViewer:
    def __init__(self, dataset: Dataset):
        gui.Application.instance.initialize()
        self.frame_counter = 0
        window = gui.Application.instance.create_window("img", width=640, height=480)
        widget = gui.SceneWidget()
        widget.scene = rendering.Open3DScene(window.renderer)
        window.add_child(widget)

        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.dataset = dataset
        self.centre = np.array(dataset.aabb[0]) + (np.array(dataset.aabb[1]) - np.array(dataset.aabb[0])) / 2
        self.cameras = dataset.draw_cameras()
        self.lines = self.cameras[0]
        self.pcd = get_frame(dataset, 0)
        widget.scene.add_geometry("camera", self.lines, mat)
        widget.scene.add_geometry("pcd", self.pcd, mat)
        self.widget = widget
        self.window = window
        self.mat = mat
        # widget.scene.camera.look_at([0, 0, 0], [1, 1, 1], [0, 0, 1])
        widget.scene.camera.look_at([0,0,0], self.centre, [0, 0, 1])
        self.initial_camera_position = [2 * x for x in self.centre]  # Start twice as far from the center
        widget.scene.camera.look_at(self.initial_camera_position, self.centre, [0, 0, 1])

        self.move_camera_to_center()

    def move_camera_to_center(self):
        """Smoothly moves the camera to the center."""
        steps = 50
        current_pos = np.array(self.initial_camera_position)
        target_pos = np.array(self.centre)

        for t in range(1, steps + 1):
            interpolated_pos = current_pos + (target_pos - current_pos) * (t / steps)
            self.widget.scene.camera.look_at(interpolated_pos, self.centre, [0, 0, 1])
            time.sleep(0.02)  # Adjust to control animation speed
        

    def update_geometry(self):
        self.widget.scene.clear_geometry()
        self.widget.scene.add_geometry("camera", self.lines, self.mat)
        self.widget.scene.add_geometry("pcd", self.pcd, self.mat)

    def load_next_frame(self):
        print("Loading next frame")
        # Load next frame
        self.frame_counter += 1
        if self.frame_counter >= len(self.cameras):
                self.frame_counter = 0

        self.lines = self.cameras[self.frame_counter]
        self.pcd = get_frame(self.dataset, self.frame_counter)

    def thread_main(self):
        while True:
            self.load_next_frame()
            gui.Application.instance.post_to_main_thread(
                self.window, self.update_geometry
            )
            time.sleep(0.5)
            

    def run(self):
        threading.Thread(target=self.thread_main).start()

        gui.Application.instance.run()


Open3DFrameViewer(oDataset).run()

# for i in range(len(cameras)):
# o3d.visualization.draw_geometries([cameras[i], get_frame(oDataset, i)])
