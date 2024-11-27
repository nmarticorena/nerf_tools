from dataclasses import dataclass, field
import json
from typing import List
import numpy as np
import cv2 
import os
import multiprocessing as mp
import spatialmath as sm
import trimesh
import torch
import open3d as o3d

@dataclass
class ReplicaFrame:
    transform_matrix:  List = field(default_factory=lambda: [])
    rgb: np.array = field(default_factory=lambda :np.array([]))
    depth: np.array = field(default_factory=lambda :np.array([]))
    noise_depth : np.array = field(default_factory=lambda :np.array([]))

@dataclass
class ReplicaDataset:
    fl_x : float = 600
    fl_y : float = 600
    cx : float =  599.5
    cy : float = 339.5
    h: int = 680
    w: int = 1200
    aabb: List[None] = field(default_factory=list)
    integer_depth_scale : float = 0.0
    depth_scale : float = 0.0
    frames: List[None] = field(default_factory=list)
    folder: str = "test"
    n_frames: int = 0
    frames_index: List[int] = field(default_factory=list)    
    gt_mesh_path : str = "test"

    def __init__(self, config, parent_path = "", sequence_name = "",load_gt= True, load_isdf = False, *args, **kwargs):
        """
        Config is the json inside the isdf data:
        Include info of the detph scale, fps, and camera intrinsics
        """ 
        self.fl_x = config["camera"]["fx"]
        self.fl_y = config["camera"]["fy"]
        self.cx = config["camera"]["cx"]
        self.cy = config["camera"]["cy"]
        self.h = config["camera"]["h"]
        self.w = config["camera"]["w"]
        self.gt_loaded = load_gt
        self.depth_scale = config["depth_scale"]
        self.integer_depth_scale = 1/self.depth_scale
                
        if load_gt:
            self.gt_mesh_path = f"{parent_path}" + "/" + config["gt_sdf_dir"] + "mesh.obj"
            self.gt_sdf_path = f"{parent_path}"  + "/" + config["gt_sdf_dir"] + "/1cm/sdf.npy"
            self.gt_sdf = np.load(self.gt_sdf_path)
            self.gt_stage_sdf_path = f"{parent_path}" + "/" + config["gt_sdf_dir"] + "/1cm/stage_sdf.npy"
            self.gt_stage_sdf = np.load(self.gt_stage_sdf_path)
        if load_isdf:
            self.frames_index = config["im_indices"]
            self.folder = f"{parent_path}" + "/" + config["seq_dir"]
        else:
            self.frames_index = []
            self.folder = parent_path + "/seqs/" + sequence_name+ "/"

        self.load_transforms()
        self.get_frames()

    def get_camera(self):
        return o3d.camera.PinholeCameraIntrinsic(self.w, 
                                             self.h, 
                                             self.fl_x, 
                                             self.fl_y,
                                             self.cx, 
                                             self.cy)

    def sample_o3d(self, idx, depth_trunc=5.0):
        '''
        input idx: index of sample
        '''
        assert idx>=0 and idx<self.n_frames, \
            f"sample index out of range [0,{self.n_frames}]"
        
        color = o3d.io.read_image(self.image_path[idx])
        depth = o3d.io.read_image(self.depth_path[idx])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale = 1.0/(self.depth_factor),
            depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

        pose = self.get_transforms_cv2([idx])

        return rgbd, pose

    def get_projection_matrix(self):
        return np.array([[1/self.fl_x, 0        , self.cx, 0],
                         [0        , 1/self.fl_y, self.cy, 0],
                         [0        , 0        , 1      , 0]])



    def load_transforms(self, orb = False):
        '''
        Load camera poses from file
        '''
        if orb:
            filename = f"{self.folder}orb_traj.txt" 
        else:
            filename = f"{self.folder}traj.txt"

        bounds_filename = f"{self.folder}bounds.txt"
        bounds = np.loadtxt(bounds_filename)
        __import__('pdb').set_trace()
        self.aabb = np.zeros((2,3))


        if self.gt_loaded:

            mesh = trimesh.exchange.load.load(self.gt_mesh_path, process=False)
        
        




            self.inv_bounds_transform, bounds_extends = trimesh.bounds.oriented_bounds(mesh)
            self.scene_center = mesh.bounds.mean(axis = 0)

            # self.inv_bounds_transform = torch.from_numpy(
            #     T_extent_to_scene).float().to('cuda')
            self.bounds_transform_np = np.linalg.inv(self.inv_bounds_transform)
            # self.bounds_transform = torch.from_numpy(
            #     self.bounds_transform_np).float().to('cuda')



            self.aabb[0,:] = mesh.vertices.min(axis = 0)
            self.aabb[1,:] = mesh.vertices.max(axis = 0)

        self.transforms  = np.loadtxt(filename).reshape(-1,4,4)
        # grid_range = [-1.0, 1.0]
        # range_dist = grid_range[1] - grid_range[0]
        # self.scene_scale_np = bounds_extends / (range_dist * 0.9)
        # self.scene_scale = torch.from_numpy(
        #     self.scene_scale_np).float().to('cuda')
        # self.inv_scene_scale = 1. / self.scene_scale
        
        return

    def get_frames(self):
        self.frames = []
        if self.frames_index == []:
            i = 0
            skip = 20
            with open(f"{self.folder}/associations.txt", 'r') as f:
                for l in f:
                    frame_json = {}
                    if i % skip == 0:
                        frame = ReplicaFrame()
                        info = l.split()
                        rgb_path = info[1]
                        depth_path = info[3]


                        # frame.rgb = cv2.imread(f"{self.folder}/{info[1]}")
                        # frame.depth = cv2.imread(f"{self.folder}/{info[3]}", cv2.IMREAD_ANYDEPTH)
                        frame.transform_matrix = self.transforms[i,:,:] @ sm.SE3.Rx(np.pi).A

                        frame_json["file_path"] = rgb_path

                        # Need to convert from 

                        frame_json["transform_matrix"] = frame.transform_matrix.tolist()
                        frame_json["depth_path"] = depth_path
                        self.frames.append(frame_json)
                    i += 1
        else:
            for i in self.frames_index:
                frame = ReplicaFrame()
                frame.rgb = cv2.imread(f"{self.folder}/results/rgb_{i:04d}.png")
                frame.depth = cv2.imread(f"{self.folder}/results/depth_{i:04d}.png", cv2.IMREAD_ANYDEPTH)
                frame.noise_depth = cv2.imread(f"{self.folder}/results/noise_depth_{i:04d}.png", cv2.IMREAD_ANYDEPTH)
                frame.transform_matrix = self.transforms[i,:,:]
                self.frames.append(frame)
    
    def get_transforms(self, indexs):
        return [np.array(self.transforms[i]) for i in indexs]

    def get_transforms_cv2(self,indexs=[]):
        '''
        Return the camera transforms in open cv standards
        '''
        if indexs == []: # Get all the training frames
            indexs = self.frames_index
        return self.get_transforms(indexs)
        

    def get_camera_intrinsic(self):
        return np.array([[self.fl_x, 0, self.cx],
                         [0, self.fl_y, self.cy],
                         [0, 0, 1]])

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


# Test
if __name__ == "__main__":

    parent_path = "/media/nmarticorena/iSDF_data/"
    import json
    with open(f"{parent_path}/results/iSDF/apt_3/0/config.json", 'r') as f:
        config = json.load(f)

    dataset = ReplicaDataset(config["dataset"], parent_path = parent_path)

    dataset.save("test.json")



