import open3d as o3d
import spatialmath  as sm
import numpy as np


def get_camera(intrinsic, extrinsic)-> o3d.geometry.LineSet:
    camera = o3d.geometry.LimeSet.create_camera_visualization(intrinsic, extrinsic)
    return camera



