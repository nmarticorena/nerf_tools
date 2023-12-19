from dataclasses import dataclass

@dataclass
class SamplePCD:
    skip_frames: int = 1 # Skip every n frames
    down_sample_frames: int = 1 # Downsample every n frames
    down_sample_voxel_size: float = 0.01 # Downsample voxel size
    
@dataclass
class DatasetPath:
    type: str = "nerf" # Dataset to use
    dataset_path: str = '/home/nmarticorena/Documents/tools/RLBench/'  # Path to the dataset 
    
@dataclass
class AABB:
    extra: float = 0.0 # Extra padding for the result alway positive
    pcd: SamplePCD = SamplePCD() # Sample Point cloud parameters
    dataset: DatasetPath = DatasetPath() # Dataset parameters
    save: bool = True # Save the aabb to the dataset 