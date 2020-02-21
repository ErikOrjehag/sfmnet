import multiprocessing

import torch
import numpy as np
import utils
import argparse
import sfm_loss
import networks.architectures
import cv2
import time
import random
import reconstruction
import options
import viz
import OpenGL.GL as gl
import pangolin as pango

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
      "net",
      "loss",
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
  
  # Window
  pango.CreateWindowAndBind('Main', int(640*(3/2)), int(480*(3/2)))
  gl.glEnable(gl.GL_DEPTH_TEST)

  # Define Projection and initial ModelView matrix
  scam = pango.OpenGlRenderState(
    pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
    pango.ModelViewLookAt(0, -0.5, -0.5, 0, 0, 1, 0, -1, 0))
  handler = pango.Handler3D(scam)

  # Create Interactive View in window
  dcam = pango.CreateDisplay()
  dcam.SetBounds(
    pango.Attach(0), 
    pango.Attach(1), 
    pango.Attach(0), 
    pango.Attach(1), -640.0/480.0)
  dcam.SetHandler(handler)

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
    for i in range(args.batch):
      print(list(data["pose"][i,0,:].cpu().detach().numpy()))
      print(list(data["pose"][i,1,:].cpu().detach().numpy()))
      print("--")

    depth_img = viz.tensor2depthimg(torch.cat((*data["depth"][0][:,0],), dim=0))
    tgt_img = viz.tensor2img(torch.cat((*data["tgt"],), dim=1))      
    img = np.concatenate((tgt_img, depth_img), axis=1)

    warp_imgs = []
    #diff_imgs = []
    for warp, diff in zip(data["warp"], data["diff"]):
      warp = restack(restack(warp, 1, -1), 0, -2)
      diff = restack(restack(diff, 1, -1), 0, -2)
      warp_imgs.append(viz.tensor2img(warp))
      #diff_imgs.append(viz.tensor2diffimg(diff))

    world = reconstruction.depth_to_3d_points(data["depth"][0], data["K"])
    points = world[0,:].view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64)
    colors = (data["tgt"][0,:].view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64) + 1) / 2

    loop = True
    while loop:
      key = cv2.waitKey(10)
      if key == 27 or pango.ShouldQuit():
        exit()
      elif key != -1:
        loop = False
      cv2.imshow("target and depth", img)
      #for i, (warp, diff) in enumerate(zip(warp_imgs, diff_imgs)):
      for i, warp in enumerate(warp_imgs):
        cv2.imshow("warp scale: %d" % i, warp)
        #cv2.imshow("diff scale: %d" % i, diff)

      gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
      gl.glClearColor(1.0, 1.0, 1.0, 1.0)
      dcam.Activate(scam)
      gl.glPointSize(5)
      pango.DrawPoints(points, colors)
      pose = np.identity(4)
      pose[:3, 3] = 0
      gl.glLineWidth(1)
      gl.glColor3f(0.0, 0.0, 1.0)
      pango.DrawCamera(pose, 0.5, 0.75, 0.8)
      pango.FinishFrame()
      

  utils.iterate_loader(args, test_loader, step_fn)
  

if __name__ == "__main__":
  multiprocessing.set_start_method('spawn', True)
  main()