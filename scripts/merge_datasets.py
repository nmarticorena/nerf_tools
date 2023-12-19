'''
Merge datasets: 

This script merge the datasets from the different sources into a single dataset.

'''


import tyro
from dataclasses import dataclass
import json
from typing import Tuple
import pathlib
import shutil
import os

@dataclass
class Args():
    folders: Tuple[str, ...]
    target_folder: str
    def __post_init__(self):
        self.target_folder = pathlib.Path(self.target_folder)
        self.target_folder.mkdir(parents=True, exist_ok=True)
    
args = tyro.cli(Args)

class DatasetMerger():
    def __init__(self, folders: Tuple[str, ...], target_folder:str):
        self.folders = [pathlib.Path(folder) for folder in folders]
        self.target_folder = pathlib.Path(target_folder)
        self.dataset = {}
        self.num_merged = 0
        
        
    def copy_images(self, folder, name):
        images = os.listdir(folder)
        images = [i for i in images if i.endswith(".png")]
        print(images)
        for i in images:
            shutil.copy(folder / i, self.target_folder / f"{name}{i}")
        
        
    def concat_frames(self, transforms, name):
        frames = transforms["frames"]
        for frame in frames:
            frame["file_path"] = f"{name}{frame['file_path']}"
            frame["depth_path"] = f"{name}{frame['depth_path']}"
        self.dataset["frames"].extend(frames)
    
    
    def concat(self, folder):
        if self.num_merged == 0:
            self.dataset = json.load(open(folder / "transforms.json"))
            self.num_merged += 1
            self.copy_images(folder, "")
        else:
            transforms = json.load(open(folder / "transforms.json"))
            self.copy_images(folder, f"{self.num_merged}_")
            self.concat_frames(transforms, f"{self.num_merged}_")
            self.num_merged += 1        
    
    def process(self):
        for folder in self.folders:
            self.concat(folder)
        json.dump(self.dataset, open(self.target_folder / "transforms.json", "w"), indent=4)
        print(f"Dataset merged into {self.target_folder}")
        return       
    
if __name__ == "__main__":
    merger = DatasetMerger(args.folders, args.target_folder)
    merger.process() 