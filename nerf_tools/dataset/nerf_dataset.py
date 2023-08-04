from dataclasses import dataclass, field
import json
from typing import List
import numpy as np
import cv2 
import os
import multiprocessing as mp
import spatialmath as sm
import open3d

@dataclass
class NeRFFrame:
    transform_matrix:  List = field(default_factory=lambda: [])
    rgb: np.array = np.array([])
    depth: np.array = np.array([])


@dataclass
class NeRFDataset:
    fl_x : float = 703.3542416031569
    fl_y : float = 703.3542416031569
    cx : float = 256
    cy : float = 256
    h: int = 0
    w: int = 0
    aabb: List[None] = field(default_factory=list)
    integer_depth_scale : float = 0.0
    depth_scale : float = 0.0
    frames: List[None] = field(default_factory=list)
    folder: str = "test"
    ros: bool = False
    n_frames: int = 0
    n_views: int = 0
    gt_mesh_path : str = ""
    frames_index : List[None] = field(default_factory=list)

    def __post_init__(self, *args, **kwargs):
        self.n_frames = 0
        os.makedirs(self.folder, exist_ok=True)

    def get_camera(self):
        return open3d.camera.PinholeCameraIntrinsic(self.w, 
                                             self.h, 
                                             self.fl_x, 
                                             self.fl_y,
                                             self.cx, 
                                             self.cy)

    def sample_o3d(self, idx, depth_trunc=10.0):
        '''
        input idx: index of sample
        '''
        assert idx>=0 and idx<len(self.frames), \
            f"sample index out of range [0,{len(self.frames)}]"
        if "png" not in self.frames[idx]['file_path']:
            img_path = self.frames[idx]['file_path'] + '.png' 
        else:
            img_path = self.frames[idx]['file_path']
        color = cv2.imread(self.folder + '/' + img_path)

        # Convert the image to a numpy array
        # image_array = color.to_numpy_array()

        # Convert RGB to BGR format using OpenCV
        bgr_image_array = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Convert the BGR image array back to an Open3D image
        color = open3d.geometry.Image(bgr_image_array)

        depth = open3d.io.read_image(self.folder + '/' + self.frames[idx]['depth_path'])
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale = 1.0/(self.integer_depth_scale),
            depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

        pose = self.get_transforms_cv2([idx])

        return rgbd, pose[0]

    def set_frames_index(self, indexs):
        self.frames_index = indexs

    def get_transforms(self, indexs):
        return [np.array(self.frames[i]["transform_matrix"]) for i in indexs]


    def get_transforms_cv2(self,indexs=[]):
        '''
        Return the camera transforms in open cv standards
        '''
        if indexs == []:
            indexs = range(len(self.frames))
        transforms = self.get_transforms(indexs)
        transforms_cv2 = []
        for transform in transforms:
            transforms_cv2.append(transform @ sm.SE3.Rx(np.pi, unit='rad').A)
        return transforms_cv2 

    def get_camera_intrinsic(self):
        return np.array([[self.fl_x, 0, self.cx],
                         [0, self.fl_y, self.cy],
                         [0, 0, 1]])

    def get_projection_matrix(self):
        return np.array([[1/self.fl_x, 0        , self.cx, 0],
                         [0        , 1/self.fl_y, self.cy, 0],
                         [0        , 0        , 1      , 0]])

    def add_frame(self, frame: NeRFFrame):
        frame_json = {}
        frame_json["transform_matrix"] = frame.transform_matrix.tolist()

        # Save rgb img
        rgb_filename = f"{self.folder}/rgb_{self.n_frames:04d}.png"
        if self.ros:
            cv2.imwrite(rgb_filename, cv2.cvtColor(
                        np.uint8(frame.rgb), cv2.COLOR_BGRA2RGBA))
        else:
            cv2.imwrite(rgb_filename, frame.rgb)

        # Save depth img
        # First convert to uint16:
        depth_uint16 = (frame.depth * 1/self.integer_depth_scale).astype(np.uint16) 
        depth_filename = f"{self.folder}/depth_{self.n_frames:04d}.png"
        cv2.imwrite(depth_filename, 
                               depth_uint16, [cv2.CV_16UC1, cv2.CV_16UC1])

        frame_json["file_path"] = rgb_filename
        frame_json["depth_path"] = depth_filename



        self.frames.append(frame_json)
        self.n_frames += 1

    def save(self, filename):

        nerf_json = {}
        nerf_json['fl_x'] = self.fl_x
        nerf_json['fl_y'] = self.fl_y
        nerf_json['cx'] = self.cx
        nerf_json['cy'] = self.cy
        nerf_json['h'] = self.h
        nerf_json['w'] = self.w

        if self.aabb is None:
            nerf_json['aabb'] = self.aabb        
        if self.integer_depth_scale != 0.0:
            nerf_json['integer_depth_scale'] = self.integer_depth_scale
        
        nerf_json['frames'] = self.frames
        with open(filename, 'w') as f:
            json.dump(nerf_json, f, indent=4)
        return






