# Load an replica dataset and save them on the nerf standard
import json
from nerf_tools.dataset.replicaCAD_dataset import ReplicaDataset as Dataset

config_filename = "/media/nmarticorena/DATA/iSDF_data/seqs/replicaCAD_info.json"
parent_path = "/media/nmarticorena/DATA/iSDF_data"


with open(config_filename, 'r') as f:
    config = json.load(f)

oDataset = Dataset(config, sequence_name="apt_2_nav", parent_path = parent_path, load_gt= False)
oDataset.save(f"{parent_path}/seqs/apt_2_nav/transforms.json")
