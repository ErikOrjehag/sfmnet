import multiprocessing

import open3d as o3d
import torch
import numpy as np
import utils
import argparse
import sfm_loss
import networks.architectures
import cv2
import time
import random
import inverse_warp
import options
import mayavi.mlab as mlab
import test_mayavi
import viz

def restack(tensor, from_dim, to_dim):
  return torch.cat(torch.split(tensor, 1, dim=from_dim), dim=to_dim).squeeze(from_dim)

def main():
  # Parse arguments
  args = options.get_args(
    description="Debug a network",
    options=[
      "batch", 
      "workers", 
      "device", 
      "load", 
      "smooth-weight", 
      "explain-weight", 
      "net",
    ])

  if args.load == "":
    print("No model file specified to load!")
    exit()

  # Construct datasets
  random.seed(1337)
  _, _, test_loader = utils.get_kitti_split(args.batch, args.workers)

  # The model
  model = networks.architectures.get_net(args)
  loss_fn = sfm_loss.get_loss_fn(args)
  
  # Load
  checkpoint = torch.load(args.load, map_location=torch.device(args.device))
  model.load_state_dict(checkpoint["model"])
  
  fig = mlab.figure(figure=None, bgcolor=(0,0,0),
    fgcolor=None, engine=None, size=(1000, 500))

  # Test
  model.train()
  model = model.to(args.device)

  def step_fn(step, inputs):

    # Forward pass and loss
    with torch.no_grad():
      loss, data = utils.forward_pass(model, loss_fn, inputs)

    print("loss %f" % loss.item())

    print(data.keys())

    print(data["pose"].shape)
    for i in range(4):
      print(list(data["pose"][i,0,:].cpu().detach().numpy()))
      print(list(data["pose"][i,1,:].cpu().detach().numpy()))
      print("--")

    depth_img = viz.tensor2depthimg(torch.cat((*data["depth"][0][:,0],), dim=0))
    tgt_img = viz.tensor2img(torch.cat((*data["tgt"],), dim=1))      
    img = np.concatenate((tgt_img, depth_img), axis=1)

    warp_imgs = []
    diff_imgs = []
    for warp, diff in zip(data["warp"], data["diff"]):
      warp = restack(restack(warp, 1, -1), 0, -2)
      diff = restack(restack(diff, 1, -1), 0, -2)
      warp_imgs.append(viz.tensor2img(warp))
      diff_imgs.append(viz.tensor2diffimg(diff))

    world = inverse_warp.depth_to_3d_points(data["depth"][0], data["K"])
    points = world[0,:].view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64)
    colors = (data["tgt"][0,:].view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64) + 1) / 2

    test_mayavi.draw_rgb_points(fig, points, colors)

    loop = True
    while loop:
      key = cv2.waitKey(10)
      if key == 27:
        exit()
      elif key != -1:
        loop = False
      cv2.imshow("target and depth", img)
      for i, (warp, diff) in enumerate(zip(warp_imgs, diff_imgs)):
        cv2.imshow("warp scale: %d" % i, warp)
        cv2.imshow("diff scale: %d" % i, diff)
      mlab.show(10)

  utils.iterate_loader(args, test_loader, step_fn)
  

if __name__ == "__main__":
  multiprocessing.set_start_method('spawn', True)
  main()