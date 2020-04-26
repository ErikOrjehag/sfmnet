
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
import geometry

def to_points_3d(img, depth, K, remove=1.0):
    m = torch.max(depth)
    mask = (depth < remove*m).squeeze(0).squeeze(0)
    world = reconstruction.depth_to_3d_points(depth, K)
    points = world[0,:,mask].view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64)
    colors = img[:,mask].view(3,-1).transpose(1,0).cpu().detach().numpy().astype(np.float64)
    return points, colors

class Debugger():

    def __init__(self, args):
        super().__init__()

        self.DEVICE = args.device

        self.loader = data.get_loader(args)

        self.model = networks.architectures.get_net(args)

        self.loss_fn = sfm_loss.get_loss_fn(args)

        checkpoint = torch.load(args.load, map_location=torch.device(args.device))
        self.model.load_state_dict(checkpoint["model"])

        self.positions = []
        self.positions_gt = []
        self.scale = None
        self.prev_tgt_i = 0

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
            #print("pose %d -> x: %.6f, y: %.6f, z: %.6f, rx: %.6f, ry: %.6f, rz: %.6f" % (i, *pose))
        
        poses = data["pose"]
        T0 = utils.torch_to_numpy(geometry.to_homog_matrix(geometry.pose_vec2mat(poses[:,1])).squeeze(0))
        T1 = np.identity(4)
        T1[:3, 3] = 0
        T2 = utils.torch_to_numpy(geometry.to_homog_matrix(geometry.pose_vec2mat(poses[:,0])).squeeze(0))

        T_gt = utils.torch_to_numpy(data["T"].squeeze(0))
        T0_gt = T_gt[0]
        T1_gt = np.identity(4)
        T1_gt[:3, 3] = 0
        T2_gt = T_gt[1]


        Ta, Tb, Tc = T0.copy(), T1.copy(), T2.copy()
        Ta_gt, Tb_gt, Tc_gt = T0_gt.copy(), T1_gt.copy(), T2_gt.copy()
        
        # Trajectory
        if self.prev_tgt_i != data["tgt_i"] - 1 or self.scale is None:
            self.positions = [] # New sequence!
            self.positions_gt = []
        self.scale = np.linalg.norm(Tc_gt[:3,-1] - Ta_gt[:3,-1]) / np.linalg.norm(Tc[:3,-1] - Ta[:3,-1])
        self.prev_tgt_i = data["tgt_i"]

        Ta_gt[:3,-1] /= self.scale
        Tc_gt[:3,-1] /= self.scale

        print(Tc_gt)
        print(Tc)
        
        if len(self.positions) == 0:
            self.positions = [Ta, Tb, Tc]
            self.positions_gt = [Ta_gt, Tb_gt, Tc_gt]
        else:
            inv = np.linalg.pinv(self.positions[-1])
            self.positions = [inv@T for T in self.positions]
            self.positions.append(Tc)

            inv_gt = np.linalg.pinv(self.positions_gt[-1])
            self.positions_gt = [inv_gt@T_gt for T_gt in self.positions_gt]
            self.positions_gt.append(Tc_gt)

        # Debug images
        depth_img = viz.tensor2depthimg(data["depth"][0][0,0])
        tgt_img = viz.tensor2img(data["tgt"][0])
        img = np.concatenate((tgt_img, depth_img), axis=1)
        tgtrefs = viz.tensor2img(torch.cat((data["refs"][0,0], data["tgt"][0], data["refs"][0,1]), dim=1))

        points, colors = to_points_3d(data["tgt"][0], data["depth"][0], data["K"])

        loop = True
        while loop:
            key = cv2.waitKey(10)
            if key == 27 or self.renderer.should_quit():
                exit()
            elif key != -1:
                loop = False
            
            cv2.imshow("target and depth", img)
            cv2.imshow("target and refs", tgtrefs)

            self.renderer.clear_screen()
            self.renderer.draw_points(points, colors)
            line = [T[:3,3] for T in self.positions]
            line_gt = [T[:3,3] for T in self.positions_gt]
            self.renderer.draw_line(line, color=(1.,0.,0.))
            self.renderer.draw_line(line_gt, color=(0.,1.,0.))
            #self.renderer.draw_cameras([T0], color=(1.,0.,0.))
            #self.renderer.draw_cameras([T1], color=(0.,1.,0.))
            #self.renderer.draw_cameras([T2], color=(0.,0.,1.))
            self.renderer.finish_frame()