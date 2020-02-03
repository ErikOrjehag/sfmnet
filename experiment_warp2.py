
import sys
import math
import time
import numpy as np
import open3d as o3d

import IO
import torch
from sequence_dataset import SequenceDataset
import cv2
import inverse_warp
import viz

def main():
    dataset = SequenceDataset(sys.argv[1])

    vis = o3d.Visualizer()
    vis.create_window()
    point_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud)

    for i in range(0, len(dataset)):
        tgt, refs, K, Kinv, sparse, dense, tgt_pose, ref_pose = dataset[i]
        
        print(tgt_pose)
        exit()

        pose_b = pose.unsqueeze(0)
        dense_b = dense.unsqueeze(0).unsqueeze(0)
        ref_b = refs[1].unsqueeze(0)
        K_b = K.unsqueeze(0)

        warp, valid, world, projpixel, grid = inverse_warp.inverse_warp(ref_b, dense_b, pose_b, K_b)

        # Vizualisation
        colors = (ref_b[0,:].view(3,-1).transpose(1,0) + 1) / 2
        points = world[0,:3].view(3,-1).transpose(1,0)
        point_cloud.points = o3d.open3d.Vector3dVector(points)
        point_cloud.colors = o3d.open3d.Vector3dVector(colors)
        vis.add_geometry(point_cloud)

        img = torch.cat((refs[0], tgt, refs[1]), dim=1)

        while cv2.waitKey(10) != 27:
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
            cv2.imshow("img", viz.tensor2img(img))
            cv2.imshow("dense", viz.tensor2depthimg(dense))
            cv2.imshow("warp", viz.tensor2img(warp.squeeze(0)))

if __name__ == "__main__":
    main()
