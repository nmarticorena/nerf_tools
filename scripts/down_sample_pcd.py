'''
@file down_sample_pcd.py

Script that down sample a point cloud and save it, using the voxelization 
down sample algorithm of open3D
'''
import open3d as o3d
import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('--folder', help = "folder with the point cloud to resample")
argparser.add_argument('-i', help= "index", type= int)
argparser.add_argument('-a', '--all', action = 'store_true')
argparser.add_argument('-s', '--voxel_size', type = float, default = 0.01,
                       help = "Voxel size used for the down sample")

args = argparser.parse_args()


folder = args.folder
if args.all:
    for i in range(1,100):
        path = folder + f"scene_pcd{i:04d}.pcd"
        pcd = o3d.io.read_point_cloud(path)

        downpcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

        print(f"Scene:{i}")
        print(f"Size Before down sample {np.asarray(pcd)}, after {np.asarray(downpcd)}")
        out_filename = folder + f"scene_down_pcd{i:04d}.pcd"

        o3d.io.write_point_cloud(out_filename, downpcd)
else:
    path = folder + f"scene_pcd{args.i:04d}.pcd"
    pcd = o3d.io.read_point_cloud(path)

    print(f"Size after down sample {np.asarray(pcd)}")
    o3d.visualization.draw_geometries([pcd])


    downpcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)

    print(f"Size after down sample {np.asarray(downpcd)}")

    o3d.visualization.draw_geometries([downpcd])

    filename = folder + f"scene_down_pcd{args.i:04d}.pcd"
