import multiprocessing

import open3d as o3d
import torch
import numpy as np
import utils
import argparse
from sfm_loss import SfmLoss
from sfm_net import SfmNet
import cv2
import time
import random
import inverse_warp
import viz

def restack(tensor, from_dim, to_dim):
  return torch.cat(torch.split(tensor, 1, dim=from_dim), dim=to_dim).squeeze(from_dim)

def main():
  # Parse arguments
  parser = argparse.ArgumentParser(description="Train and eval sfm nets.")
  parser.add_argument("--name", default="model", type=str, help="The run name.")
  parser.add_argument("--batch", default=4, type=int, help="The batch size.")
  parser.add_argument("--workers", default=1, type=int, help="The number of worker threads.")
  parser.add_argument("--device", default="cuda", type=str, help="The device to run on cpu/cuda.")
  parser.add_argument("--load", default="", type=str, help="Load state file.")
  args = parser.parse_args()
  print("\nCurrent arguments -> ", args, "\n")

  if args.device == "cuda" and not torch.cuda.is_available():
    print("CUDA is not available!")
    exit()

  if args.load == "":
    print("No model file specified to load!")
    exit()

  # Construct datasets
  random.seed(1337)
  _, _, test_loader = utils.get_kitti_split(args.batch, args.workers)

  # The model
  model = SfmNet()
  criterion = SfmLoss()
  
  # Load
  checkpoint = torch.load(args.load, map_location=torch.device(args.device))
  model.load_state_dict(checkpoint["model"])
  
  vis = o3d.Visualizer()
  vis.create_window()
  point_cloud = o3d.geometry.PointCloud()
  vis.add_geometry(point_cloud)

  # Test
  model.train()
  model = model.to(args.device)

  for step, inputs in enumerate(test_loader):

    # Send inputs to device
    inputs = [inp.to(args.device) for inp in inputs]

    tgt, refs, K = inputs[:3]

    # Forward pass and loss
    with torch.no_grad():
      outputs = model(inputs)
      (depths), (pose, exp) = outputs
      loss, warps, diffs = criterion(inputs, outputs)

      print("loss %f" % loss.item())

      print(pose.shape)
      for i in range(4):
        print(list(pose[i,0,:].cpu().detach().numpy()))
        print(list(pose[i,1,:].cpu().detach().numpy()))
        print("--")

      depth_img = viz.tensor2depthimg(torch.cat((*depths[0][:,0],), dim=0))
      tgt_img = viz.tensor2img(torch.cat((*tgt,), dim=1))      
      img = np.concatenate((tgt_img, depth_img), axis=1)

      warp_imgs = []
      diff_imgs = []
      for warp, diff in zip(warps, diffs):
        warp = restack(restack(warp, 1, -1), 0, -2)
        diff = restack(restack(diff, 1, -1), 0, -2)
        warp_imgs.append(viz.tensor2img(warp))
        diff_imgs.append(viz.tensor2diffimg(diff))

      world = inverse_warp.depth_to_3d_points(depths[0], K)
      points = world[0,:].view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64)
      colors = (tgt[0,:].view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64) + 1) / 2
    
    point_cloud.points = o3d.open3d.Vector3dVector(points)
    point_cloud.colors = o3d.open3d.Vector3dVector(colors)
    vis.add_geometry(point_cloud)

    while cv2.waitKey(10) != 27:
      #o3d.visualization.draw_geometries([point_cloud])
      vis.update_geometry()
      vis.poll_events()
      vis.update_renderer()
      cv2.imshow("target and depth", img)
      for i, (warp, diff) in enumerate(zip(warp_imgs, diff_imgs)):
        cv2.imshow("warp scale: %d" % i, warp)
        cv2.imshow("diff scale: %d" % i, diff)
  

if __name__ == "__main__":
  multiprocessing.set_start_method('spawn', True)
  main()