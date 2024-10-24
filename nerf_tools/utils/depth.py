import open3d as o3d
import numpy as np

from nerf_tools.dataset.nerf_dataset import NeRFDataset
from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset
from typing import Union

timing = True
if timing:
    import time


def get_camera(intrinsic, extrinsic) -> o3d.geometry.LineSet:
    camera = o3d.geometry.LimeSet.create_camera_visualization(intrinsic, extrinsic)
    return camera


def get_frame(
    dataset: Union[NeRFDataset, ReplicaDataset], index: int
) -> o3d.geometry.LineSet:
    rgbd, pose = dataset.sample_o3d(index, depth_trunc=10.0)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, dataset.get_camera())
    pcd.transform(pose)
    return pcd


def get_pointcloud(
    dataset: Union[NeRFDataset, ReplicaDataset],
    max_depth=10.0,
    skip_frames=1,
    filter_step=5,
    voxel_size=0.05,
) -> o3d.geometry.PointCloud:
    pcd_final = o3d.geometry.PointCloud()
    camera = dataset.get_camera()
    for ix, frame in enumerate(dataset.frames):
        if ix % skip_frames == 0:
            if timing:
                start = time.time()
            rgbd, pose = dataset.sample_o3d(ix, depth_trunc=max_depth)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)

            pcd.transform(pose)

            pcd_final.points = o3d.utility.Vector3dVector(
                np.concatenate(
                    [np.asarray(pcd_final.points), np.asarray(pcd.points)], axis=0
                )
            )
            pcd_final.colors = o3d.utility.Vector3dVector(
                np.concatenate(
                    [np.asarray(pcd_final.colors), np.asarray(pcd.colors)], axis=0
                )
            )
            if timing:
                end = time.time()
                print(f"Frame {ix} took {end - start} seconds")
        # downsample the point cloud
        if ix % (filter_step * skip_frames) == 0:
            if timing:
                start = time.time()
            pcd_final = pcd_final.voxel_down_sample(voxel_size=voxel_size)
            if timing:
                end = time.time()
                print(f"Downsample took {end - start} seconds")
    pcd_final = pcd_final.voxel_down_sample(voxel_size=voxel_size)
    return pcd_final


def get_tsdf(dataset: Union[NeRFDataset, ReplicaDataset], depth_trunc=10):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    camera = dataset.get_camera()
    for ix, frame in enumerate(dataset.frames):
        rgbd, pose = dataset.sample_o3d(ix, depth_trunc=depth_trunc)

        volume.integrate(rgbd, camera, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh
