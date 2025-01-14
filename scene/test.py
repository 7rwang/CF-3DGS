

import os
from scene.dataset_readers import readColmapCameras
from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary

path = "/home/xduo/桌面/CF-3DGS/data/car_4v"
cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

print(cam_extrinsics) 