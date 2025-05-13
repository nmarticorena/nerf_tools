from dataclasses import dataclass, field


@dataclass
class SamplePCD:
    skip_frames: int = 1  # Skip every n frames
    down_sample_frames: int = 1  # Downsample every n frames
    down_sample_voxel_size: float = 0.01  # Downsample voxel size
    max_depth: float = 10.0  # Max depth to sample


@dataclass
class DatasetPath:
    type: str = "nerf"  # Dataset to use
    "Dataset type, nerf or replica"
    dataset_path: str = (
        "/media/nmarticorena/DATA/datasets/NeRFCapture/cupboard"  # Path to the dataset
    )
    "Full path to the dataset"


@dataclass
class AABB:
    extra: float = 0.0  # Extra padding for the result alway positive
    pcd: SamplePCD = field(default_factory=lambda :SamplePCD())
    dataset: DatasetPath = field(default_factory=lambda :DatasetPath())  # Dataset parameters
    save: bool = True  # Save the aabb to the dataset
    gui: bool = True
    web: bool = False # to use the webrtc open3d


@dataclass
class TSDF:
    dataset: DatasetPath = field(default_factory=lambda :DatasetPath())
    "Dataset to use, specifies the type and path"
    gui: bool = True
    "Visualize result"
    save: bool = True
    "Save the mesh to the dataset"
    name: str = "mesh"
    "Name of the mesh to save"
    depth_trunc: float = 10  # Max depth
    "Max depth to sample"
    voxel_size: float =  4.0/512.0
    "Voxel size for the tsdf integration"


@dataclass
class Bias:
    dataset: DatasetPath = field(default_factory=lambda :DatasetPath())
    new_folder: str = ""
    distance: float = 0.25


@dataclass
class Mask:
    dataset: DatasetPath = field(default_factory=lambda :DatasetPath())
    new_folder: str = ""
    masked_folder: str = ""
    inv: bool = False
