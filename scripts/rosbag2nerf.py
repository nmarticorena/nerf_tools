"""
@author: Nicolas Marticorena

This scripts takes as input a extracted rosbag using the
https://github.com/qcr/rosdata package and convert it to NeRF dataset standard


This include the change on the transformation of the camera to the blender standard
https://i.stack.imgur.com/Fq66R.png
"""

import json
import os
import pandas as pd
import yaml


import numpy as np

import spatialmath as sm
import spatialmath.base as smb


class rosbag2nerf:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.json_dict = {}
        self.get_config(os.path.join(folder_name, "data_files/camera_info.yaml"))
        self.get_transforms(os.path.join(folder_name, "data_files/camera_poses.csv"))

    def get_config(self, path):
        """
        Get the config of the camera from the rosbag
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        self.json_dict["fl_x"] = data["K"][0]
        self.json_dict["fl_y"] = data["K"][4]
        self.json_dict["w"] = data["width"]
        self.json_dict["h"] = data["height"]
        self.json_dict["cx"] = data["K"][2]
        self.json_dict["cy"] = data["K"][5]
        self.json_dict["integer_depth_scale"] = 1 / 5000
        self.json_dict["aabb"] = []
        self.json_dict["frames"] = []
        self.initial = 0

        return data

    def get_homogenous_transformation(self, translation, rpy):
        # T = np.zeros((4, 4))
        # T[:-1, -1] = translation
        #
        # # T[0,-1] = translation[1]
        # # T[1,-1] = translation[2]
        # # T[2,-1] = translation[0]
        #
        # M = np.zeros((4, 4))
        # M[-1, -1] = 1
        # M += T
        #
        # rpy = np.array(rpy)
        #
        # rpy[0] = -1.0 * rpy[0]
        #
        # Rotation = R.from_euler("xyz", rpy)
        # M[:-1, :-1] = Rotation.as_matrix()
        #
        T = sm.SE3.Rt(R=smb.q2r(rpy, order="xyzs"), t=translation, check=False)
        T = T.norm()
        return T

    def get_transforms(self, path):
        """
        Get the transform of the camera from the rosbag
        """
        df = pd.read_csv(path)

        frames = 0
        x = []
        y = []
        z = []
        for index, row in df.iterrows():
            # if index % 10 != 0:
            #     continue
            # if frames > (self.json_dict["n_frames"]):
            #     break
            # if index < self.initial:
            #     continue

            frames += 1
            # Get the transformation
            # translation = sm.SE3(row["pos_x"], row["pos_y"], row["pos_z"])
            # # rotation = sm.UnitQuaternion(row['quat_w'], [row['quat_x'], row['quat_y'], row['quat_z']])
            #
            # rotation = R.from_quat(
            #     [row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"]]
            # )

            # eul_rotation = rotation.as_euler("xyz")
            x.append(row["pos_x"])
            y.append(row["pos_y"])
            z.append(row["pos_z"])
            # # Get the transformation to blender standard
            # transformation = self.get_homogenous_transformation(
            #     translation.t, eul_rotation
            # )
            t = np.array([row["pos_x"], row["pos_y"], row["pos_z"]])
            q = np.array([row["quat_x"], row["quat_y"], row["quat_z"], row["quat_w"]])

            transformation = self.get_homogenous_transformation(t, q) * sm.SE3.Rx(np.pi)

            # Check if there is images
            if os.path.exists(
                os.path.join(self.folder_name, f"images/image_{index}.jpeg")
            ):
                if os.path.exists(
                    os.path.join(self.folder_name, f"images/depth_{index}.png")
                ):
                    observation_json = json.dumps(
                        {
                            "file_path": f"images/image_{index}.jpeg",
                            "transform_matrix": transformation.A.tolist(),
                            "depth_path": f"images/depth_{index}.png",
                        }
                    )
                    self.json_dict["frames"].append(json.loads(observation_json))

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        offset = 1.0
        self.json_dict["aabb"] = [
            [np.min(x) - offset, np.min(y) - offset, np.min(z) - offset],
            [np.max(x) + offset, np.max(y) + offset, np.max(z) + offset],
        ]

        with open(os.path.join(self.folder_name, "transforms.json"), "w") as f:
            json.dump(self.json_dict, f, indent=4)


if __name__ == "__main__":
    import tyro
    from dataclasses import dataclass

    @dataclass
    class Args:
        folder_name: str
        "Folder name of the data folder extracted"

    args = tyro.cli(Args)

    rosbag2nerf(args.folder_name)
