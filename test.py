
import numpy as np
import open3d as o3d

import IO
import torch
from sequence_dataset import tensor_imshow, tensor_depthshow
import cv2
import inverse_warp

disp, scale = IO.readPFM("/home/ai/Data/FlyingThings3D_subset/train/disparity/left/0000000.pfm")
image = IO.readImage("/home/ai/Data/FlyingThings3D_subset/train/image_clean/left/0000000.png")
disp = torch.from_numpy(np.ascontiguousarray(disp)).unsqueeze(0)
image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0) / 255.0

f = 1050.0
K = torch.tensor([
    [f,   0.0, 479.5],
    [0.0,   f, 269.5],
    [0.0, 0.0,   1.0]
]).unsqueeze(0)

pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unsqueeze(0)
#depth = torch.ones((image.shape[-2], image.shape[-1])).type_as(image).unsqueeze(0)
depth = f * 1.0 / -disp

print(disp.shape)
print(scale)
print(image.shape)
print(depth.shape)

warped, valid, pixels, rays, world = inverse_warp.inverse_warp(image, depth, pose, K)

points = np.random.rand(10000, 3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([point_cloud])

if False:
    tensor_depthshow("depth", depth.squeeze(0))
    tensor_imshow("image", image.squeeze(0))
    tensor_imshow("warped", warped.squeeze(0))
    cv2.waitKey(0)