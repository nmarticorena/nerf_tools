import open3d as o3d
from nerf_tools.dataset.nerf_dataset import NeRFDataset, load_from_json
from nerf_tools.utils.depth import get_pointcloud

from nerf_tools.configs import DatasetPath, SamplePCD
import os


def load_dataset(dataset_config: DatasetPath) -> NeRFDataset:
    if ".json" not in dataset_config.dataset_path:
        json_path = os.path.join(
            dataset_config.dataset_path, "transforms.json")
    else:
        json_path = dataset_config.dataset_path
    oDataset = load_from_json(json_path)
    camera_index = range(0, len(oDataset.frames))
    oDataset.set_frames_index(camera_index)
    return oDataset


def load_pcd(dataset_path, pcd_config: SamplePCD) -> o3d.geometry.PointCloud:
    dataset = load_dataset(dataset_path)
    pcd = get_pointcloud(
        dataset,
        pcd_config.max_depth,
        pcd_config.skip_frames,
        pcd_config.down_sample_frames,
        voxel_size=pcd_config.down_sample_voxel_size,
    )
    return pcd
