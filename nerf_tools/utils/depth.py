import open3d as o3d
import numpy as np
from nerf_tools.dataset.nerf_dataset import NeRFDataset
from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset
from typing import Union

def get_pointcloud(dataset: Union[NeRFDataset, ReplicaDataset], max_depth = 10) \
    -> o3d.geometry.PointCloud:

    pcd_final = o3d.geometry.PointCloud()
    camera = dataset.get_camera()    
    for ix, frame in enumerate(dataset.frames):
        rgbd, pose = dataset.sample_o3d(ix, depth_trunc= max_depth)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)

        pcd.transform(pose)

        pcd_final.points = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(pcd_final.points),np.asarray(pcd.points)],
            axis=0))
        pcd_final.colors = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(pcd_final.colors),np.asarray(pcd.colors)],
            axis=0))

    return pcd_final

def get_tsdf(dataset: Union[NeRFDataset, ReplicaDataset]):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
   
    camera = dataset.get_camera()
    for ix, frame in enumerate(dataset.frames):
        rgbd, pose = dataset.sample_o3d(ix, depth_trunc= 0.8)

        volume.integrate(
            rgbd,
            camera,
            np.linalg.inv(pose))
    return volume.extract_triangle_mesh()