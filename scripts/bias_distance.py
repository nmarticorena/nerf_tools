'''
@author: nmarticorena

This scripts genreates a copy of a dataset after removing the object of  interestes by using a semantic mask
'''

#import open3d as o3d
import numpy as np
import os
import json


import tyro

from nerf_tools.configs import *

args = tyro.cli(Bias)

os.makedirs(args.new_folder, exist_ok= True)
os.makedirs(os.path.join(args.new_folder,"images"),exist_ok=True)

if args.dataset.type == 'nerf':
    from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset
    from nerf_tools.dataset.nerf_dataset import load_from_json
    
    dataset_path = args.dataset.dataset_path
    
    if ".json" not in dataset_path:
        dataset_path = os.path.join(dataset_path, 'transforms.json')
    with open(dataset_path, "r") as f:
        config = json.load(f)
    oDataset = load_from_json(dataset_path)
    camera_index = range(0, len(oDataset.frames))
    oDataset.set_frames_index(camera_index)
else:
    exit(1)

import shutil
import matplotlib.pyplot as plt 
import cv2

shutil.copy(dataset_path, os.path.join(args.new_folder,"transforms.json"))

depth_scale = oDataset.integer_depth_scale


for frame in oDataset.frames:
    rgb_path = frame["file_path"]
    depth_path = frame["depth_path"]

    filename = rgb_path
    depth_filename = depth_path

    depth = cv2.imread(os.path.join(args.dataset.dataset_path,depth_path), -1)

    depth_mm = depth.astype(np.float32) * depth_scale
    depth_mm -= Bias.distance

    depth_new = (depth_mm * 1 / depth_scale).astype(np.uint16)
    

    plt.imshow(depth)
    plt.show()
    path_new_depth = os.path.join(args.new_folder,depth_path)
    
    # Copy the rgb images
    shutil.copy(os.path.join(args.dataset.dataset_path,rgb_path), 
                os.path.join(args.new_folder, rgb_path))
    cv2.imwrite(path_new_depth, depth_new, [cv2.CV_16UC1, cv2.CV_16UC1])
    print(path_new_depth)

    
    

    
