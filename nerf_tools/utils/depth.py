import open3d as o3d
import numpy as np

from nerf_tools.dataset.nerf_dataset import NeRFDataset
from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset
from typing import Union, Optional

timing = True
if timing:
    import time


def get_camera(intrinsic, extrinsic) -> o3d.geometry.LineSet:
    camera = o3d.geometry.LimeSet.create_camera_visualization(
        intrinsic, extrinsic)
    return camera


def get_frame(
    dataset: Union[NeRFDataset, ReplicaDataset], index: int
) -> o3d.geometry.LineSet:
    rgbd, pose = dataset.sample_o3d(index, depth_trunc=10.0)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, dataset.get_camera())
    pcd.transform(pose)
    return pcd

def estimate_normal(depth: np.ndarray,K: np.ndarray, downsample:int = 1) -> np.ndarray:
    """
    Estimate the normal of a depth image using the intrinsic camera matrix
    """
    depth_downsample = depth[::downsample, ::downsample]

    H, W = depth_downsample.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2]/downsample, K[1, 2]/downsample

    # Create grid of pixel coordinates
    x = np.arange(0, W)
    y = np.arange(0, H)
    xx, yy = np.meshgrid(x, y, indexing="xy")

    # Compute 3D coordinates
    X = (xx - cx) * depth_downsample / fx
    Y = (yy - cy) * depth_downsample / fy
    Z = depth_downsample

    # Compute vectors from neighboring pixels
    Vx = np.zeros((H, W, 3))
    Vy = np.zeros((H, W, 3))

    Vx[:, :-1, 0] = X[:, 1:] - X[:, :-1]
    Vx[:, :-1, 1] = Y[:, 1:] - Y[:, :-1]
    Vx[:, :-1, 2] = Z[:, 1:] - Z[:, :-1]

    Vy[:-1, :, 0] = X[1:, :] - X[:-1, :]
    Vy[:-1, :, 1] = Y[1:, :] - Y[:-1, :]
    Vy[:-1, :, 2] = Z[1:, :] - Z[:-1, :]

    # Compute normals using cross product
    normals = np.cross(Vx, Vy, axis=2)

    # Normalize the normals
    normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)

    # Replace NaN or infinite values
    normals = np.nan_to_num(normals)
    return normals # (H, W, 3)


def get_pointcloud(
    dataset: Union[NeRFDataset, ReplicaDataset],
    max_depth=10.0,
    skip_frames=1,
    filter_step=5,
    voxel_size=0.05,
    normal = False,
    max_points: Optional[int] = None,
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
    # Check if point clous is larger than max_points
    if max_points is not None and len(pcd_final.points) > max_points:
        print(f"Point cloud size: {len(pcd_final.points)}")
        ratio = len(pcd_final.points) / max_points
        pcd_final = pcd_final.random_down_sample(1/ratio)
        print(f"Performing a uniform downsample with a ration of {ratio}")



    if normal:
        camera_poses = dataset.get_transforms_cv2()
        translations = np.array([T[:3, 3] for T in camera_poses])
        average_translation = translations.mean(axis=0)

        pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd_final.orient_normals_towards_camera_location(average_translation)

    return pcd_final


def get_tsdf(dataset: Union[NeRFDataset, ReplicaDataset], depth_trunc: float = 10., voxel_size: float = 4.0/512.0) -> o3d.geometry.TriangleMesh:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size, sdf_trunc=0.04, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    camera = dataset.get_camera()
    for ix, frame in enumerate(dataset.frames):
        rgbd, pose = dataset.sample_o3d(ix, depth_trunc=depth_trunc)

        volume.integrate(rgbd, camera, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh
