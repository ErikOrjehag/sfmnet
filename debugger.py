
import torch
import sfm_loss
import data
import networks.architectures
import utils
import viz
import numpy as np
import cv2
import reconstruction
from renderer import Renderer

def to_points_3d(img, depth, K):
    world = reconstruction.depth_to_3d_points(depth, K)
    points = world[0,:].view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64)
    colors = (img.view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64) + 1) / 2
    return points, colors

class Debugger():

    def __init__(self, args):
        super().__init__()

        self.DEVICE = args.device

        self.loader = data.get_kitti(args.workers)

        self.model = networks.architectures.get_net(args)

        self.loss_fn = sfm_loss.get_loss_fn(args)

        checkpoint = torch.load(args.load, map_location=torch.device(args.device))
        self.model.load_state_dict(checkpoint["model"])

        self.renderer = Renderer("Main")

    def run(self):

        self.model.eval()

        utils.iterate_loader(self.DEVICE, self.loader, self._step_fn)

    def _step_fn(self, step, inputs):

        # Forward pass and loss
        with torch.no_grad():
            loss, data = utils.forward_pass(self.model, self.loss_fn, inputs)

        print(f"loss {loss.item():.3f}")
        
        for i in range(data["pose"].shape[1]):
            pose = list(data["pose"][0,i,:].cpu().detach().numpy())
            print("pose %d -> x: %.6f, y: %.6f, z: %.6f, rx: %.6f, ry: %.6f, rz: %.6f" % (i, *pose))
        

        print(data["pose"].shape)

        depth_img = viz.tensor2depthimg(data["depth"][0][0,0])
        tgt_img = viz.tensor2img(data["tgt"][0])
        img = np.concatenate((tgt_img, depth_img), axis=1)

        points, colors = to_points_3d(data["tgt"][0], data["depth"][0], data["K"])

        loop = True
        while loop:
            key = cv2.waitKey(10)
            if key == 27 or self.renderer.should_quit():
                exit()
            elif key != -1:
                loop = False
            
            cv2.imshow("target and depth", img)
            
            self.renderer.clear_screen()
            self.renderer.draw_points(points, colors)
            self.renderer.draw_camera()
            self.renderer.finish_frame()