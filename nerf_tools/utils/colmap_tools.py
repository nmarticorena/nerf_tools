import open3d as o3d
import spatialmath as sm
import numpy as np
from nerf_tools.dataset.nerf_dataset import NeRFDataset as Dataset


def point_cloud_saver(pcd: o3d.geometry.PointCloud, path: str):
    """
    Save a point cloud to a txt file using the colmap standard

    POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    """
    with open(path, "w") as f:
        for i in range(len(pcd.points)):
            x, y, z = pcd.points[i]
            r, g, b = pcd.colors[i]
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            f.write(f"{i} {x} {y} {z} {r} {g} {b} 0 0\n")


def camera_poses_saver(dataset: Dataset, path):
    """
    Save the camera poses to a txt file using the colmap standard

    IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    """
    with open(path, "w") as f:
        for i in range(len(dataset.frames)):
            frame = dataset.frames[i]

            T_WC = np.linalg.inv(dataset.get_transforms_cv2([i])[0])

            tx, ty, tz = T_WC[:3, 3]
            # before taking the rotation we need to convert them to opencv standard
            R_WC = T_WC[:3, :3]
            # R_WC = R_WC.norm()  # fix any scaling
            qw, qx, qy, qz = sm.base.r2q(
                R_WC, order="sxyz"
            )  # we pass the array to skip the check
            frame_name = frame["file_path"].replace("images/", "")
            camera_id = 1
            f.write(
                f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {frame_name}\n\n"
            )
