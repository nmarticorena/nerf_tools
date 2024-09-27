from dataclasses import dataclass, field
import json
from typing import List
import numpy as np
import cv2
import os
import multiprocessing as mp
import spatialmath as sm
import open3d

# from nerf_tools.utils.depth import *
import open3d as o3d


def get_camera(intrinsic, extrinsic) -> o3d.geometry.LineSet:
    camera = o3d.geometry.LimeSet.create_camera_visualization(intrinsic, extrinsic)
    return camera


@dataclass
class NeRFFrame:
    transform_matrix: List = field(default_factory=lambda: [])
    rgb: np.array = np.array([])
    depth: np.array = np.array([])


@dataclass
class NeRFDataset:
    fl_x: float = 703.3542416031569
    fl_y: float = 703.3542416031569
    cx: float = 256
    cy: float = 256
    h: int = 0
    w: int = 0
    aabb: List[None] = field(default_factory=list)
    integer_depth_scale: float = 0.0
    depth_scale: float = 0.0
    frames: List[None] = field(default_factory=list)
    folder: str = ""
    ros: bool = False
    n_frames: int = 0
    n_views: int = 0
    gt_mesh_path: str = ""
    frames_index: List[None] = field(default_factory=list)
    path: str = ""
    max_depth: float = 10.0

    def __str__(self):
        string = f"""NeRF Dataset
        aabb = {self.aabb}
        n_frames = {self.n_frames}
        folder = {self.folder}
        path = {self.path}
        """
        return string

    def __post_init__(self, *args, **kwargs):
        self.n_frames = 0
        os.makedirs(f"{self.path}/{self.folder}", exist_ok=True)

    def get_camera(self):
        return open3d.camera.PinholeCameraIntrinsic(
            int(self.w), int(self.h), self.fl_x, self.fl_y, self.cx, self.cy
        )

    def draw_cameras(self):
        intrinsic = self.get_camera()
        extrinsic = self.get_transforms_cv2()
        Lines = []
        for W_TC in extrinsic:
            Lines.append(
                o3d.geometry.LineSet.create_camera_visualization(
                    intrinsic, np.linalg.inv(W_TC), scale=0.1
                )
            )
        return Lines

    def sample_o3d(self, idx, depth_trunc=10.0):
        """
        input idx: index of sample
        """
        assert idx >= 0 and idx < len(
            self.frames
        ), f"sample index out of range [0,{len(self.frames)}]"
        if "." not in self.frames[idx]["file_path"]:
            img_path = self.frames[idx]["file_path"] + ".png"
        else:
            img_path = self.frames[idx]["file_path"]
        color = cv2.imread(self.path + "/" + img_path)

        # Convert the image to a numpy array
        # image_array = color.to_numpy_array()

        # Convert RGB to BGR format using OpenCV
        bgr_image_array = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Convert the BGR image array back to an Open3D image
        color = open3d.geometry.Image(bgr_image_array)

        depth = open3d.io.read_image(self.path + "/" + self.frames[idx]["depth_path"])
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0 / (self.integer_depth_scale),
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )

        pose = self.get_transforms_cv2([idx])

        return rgbd, pose[0]

    def set_frames_index(self, indexs):
        self.frames_index = indexs

    def get_transforms(self, indexs):
        return [np.array(self.frames[i]["transform_matrix"]) for i in indexs]

    def get_transforms_cv2(self, indexs=[]):
        """
        Return the camera transforms in open cv standards
        """
        if indexs == []:
            indexs = range(len(self.frames))
        transforms = self.get_transforms(indexs)
        transforms_cv2 = []
        for transform in transforms:
            transforms_cv2.append(transform @ sm.SE3.Rx(np.pi, unit="rad").A)
            # transforms_cv2.append(transform)
        return transforms_cv2

    def get_camera_intrinsic(self):
        return np.array([[self.fl_x, 0, self.cx], [0, self.fl_y, self.cy], [0, 0, 1]])

    def get_projection_matrix(self):
        return np.array(
            [
                [1 / self.fl_x, 0, self.cx, 0],
                [0, 1 / self.fl_y, self.cy, 0],
                [0, 0, 1, 0],
            ]
        )

    def add_frame(self, frame: NeRFFrame):
        frame_json = {}
        frame_json["transform_matrix"] = frame.transform_matrix.tolist()

        # Save rgb img
        rgb_filename = f"rgb_{self.n_frames:04d}.png"
        write_rgb_filename = f"{self.path}/{rgb_filename}"
        if self.ros:
            cv2.imwrite(
                write_rgb_filename,
                cv2.cvtColor(np.uint8(frame.rgb), cv2.COLOR_BGRA2RGBA),
            )
        else:
            cv2.imwrite(write_rgb_filename, frame.rgb)

        # Save depth img
        # First convert to uint16:

        masked_depth = frame.depth.copy()
        masked_depth[masked_depth > self.max_depth] = 0  # Mask the max depth
        depth_uint16 = (masked_depth * 1 / self.integer_depth_scale).astype(np.uint16)

        depth_filename = f"depth_{self.n_frames:04d}.png"

        cv2.imwrite(
            f"{self.path}/{depth_filename}", depth_uint16, [cv2.CV_16UC1, cv2.CV_16UC1]
        )

        frame_json["file_path"] = rgb_filename
        frame_json["depth_path"] = depth_filename

        self.frames.append(frame_json)
        self.n_frames += 1

    def save(self, filename):
        nerf_json = {}
        nerf_json["fl_x"] = self.fl_x
        nerf_json["fl_y"] = self.fl_y
        nerf_json["cx"] = self.cx
        nerf_json["cy"] = self.cy
        nerf_json["h"] = self.h
        nerf_json["w"] = self.w

        if self.aabb is None:
            nerf_json["aabb"] = self.aabb
        if self.integer_depth_scale != 0.0:
            nerf_json["integer_depth_scale"] = self.integer_depth_scale

        nerf_json["frames"] = self.frames
        with open(filename, "w") as f:
            json.dump(nerf_json, f, indent=4)
        return


def load_from_json(filepath: str) -> NeRFDataset:
    """Create a NeRFDataset instance from a transform.json file.

    Args:
        filepath (str): Path to transform.json file.
    Returns:
        NeRFDataset: NeRFDataset instance.
    """
    if not filepath.endswith(".json"):
        filepath = os.path.join(filepath, "transforms.json")

    with open(filepath, "r") as f:
        nerf_json = json.load(f)

    expected_keys = set(NeRFDataset.__annotations__.keys())
    filtered_data_dict = {k: v for k, v in nerf_json.items() if k in expected_keys}

    path = os.path.dirname(filepath)

    return NeRFDataset(**filtered_data_dict, path=path)


ROS = False
try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo

    ROS = True
except ImportError:
    print("ROS not installed")

if ROS:

    class RosToNeRF(NeRFDataset):
        def __init__(self, folder, camera_info: CameraInfo, max_depth=10.0):
            super().__init__(path=folder, folder="", ros=True)
            self.fl_x = camera_info.K[0]
            self.fl_y = camera_info.K[4]
            self.cx = camera_info.K[2]
            self.cy = camera_info.K[5]
            self.h = camera_info.height
            self.w = camera_info.width
            self.max_depth = max_depth
            max_np_uint = 1 << 16
            self.integer_depth_scale = self.max_depth / max_np_uint

        def record_frame(self, img: Image, depth: Image, transform: sm.SE3):
            frame = NeRFFrame()
            frame.transform_matrix = transform.A
            frame.rgb = img
            frame.depth = depth

            return super().add_frame(frame)
