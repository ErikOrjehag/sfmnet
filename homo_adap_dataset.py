
import os
import glob
import torch
import torch.utils.data as data
from imageio import imread
import utils
import homography
import numpy as np
import cv2
import reconstruction
import geometry

class HomoAdapDataset(data.Dataset):

    def __init__(self, root):
        super().__init__()

        self.images = sorted(glob.glob(os.path.join(root, "*.jpg")))

    def __len__(self):
        return len(self.images)

    def scale_crop(self, img):
        H, W = img.shape[:2]
        imH, imW = 128, 416
        scale = imW / W
        cropW = 0
        cropH = int((H * scale - imH) / 2)
        if cropH < 0:
            scale = imH / H
            cropW = int((W * scale - imW) / 2)
            cropH = 0
        img = cv2.resize(img, None, fx=scale*1.02, fy=scale*1.02, interpolation=cv2.INTER_AREA)
        img = img[cropH:cropH+imH,cropW:cropW+imW]
        return img

    def __getitem__(self, index):
        img = imread(self.images[index])
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        img = self.scale_crop(img)
        img = utils.cv2_to_torch(img)
        h = homography.random_homography(rotation=np.pi/20, translation=10, scale=0.2, sheer=0.05, projective=0.001)
        #h = homography.random_homography(rotation=(np.pi/180)*5, translation=5, scale=0.05, sheer=0.01, projective=0.0)
        warp = homography.warp_image(img, h)
        data = { 
            "img": img,
            "warp": warp,
            "homography": h,
        }
        if img.shape[1] != 128 or img.shape[2] != 416:
            print(img.shape)
            print(self.images[index])
        return data

class HomoAdapDatasetFromSequences(HomoAdapDataset):

    def __init__(self, root):

        sequences = sorted(glob.glob(os.path.join(root, "*", "")))

        self.images = []
        
        for sequence in sequences:
            self.images += sorted(glob.glob(os.path.join(sequence, "*.jpg")))

class HomoAdapDatasetCocoKittiLyft(HomoAdapDataset):

    def __init__(self, root):

        # Coco
        self.images = sorted(glob.glob(os.path.join("/home/ai/Code/Data/coco/unlabeled2017/", "*.jpg")))

        # Kitti
        sequences = sorted(glob.glob(os.path.join("/home/ai/Code/Data/kitti2", "*", "")))

        # Lyft
        sequences += sorted(glob.glob(os.path.join("/home/ai/Code/Data/lyft/v1.02-train/", "*", "")))

        for sequence in sequences:
            self.images += sorted(glob.glob(os.path.join(sequence, "*.jpg")))

class HomoAdapSynthPointDataset():

    def __init__(self, root=None):
        super().__init__()

        self.N_points = 128

        self.H = 128
        self.W = 416

    def __len__(self):
        return 5000

    def __getitem__(self, index):

        coords = torch.rand((2, self.N_points))
        coords[0] = (coords[0]-0.5)# * self.W
        coords[1] = (coords[1]-0.5)# * self.H
        #coords[0] *= self.W
        #coords[1] *= self.H
        #coords[0] = (coords[0]-0.5)*2
        #coords[1] = (coords[1]-0.5)*2
        coords = torch.cat((coords, torch.ones((1, self.N_points))), dim=0)
        
        #h = homography.random_homography(rotation=np.pi/20, translation=10, scale=0.2, sheer=0.05, projective=0.001)
        #h = homography.random_homography(rotation=0, translation=0, scale=0.9, sheer=0, projective=0)
        h = homography.random_homography(rotation=np.pi/20, translation=0.05, scale=0.2, sheer=0.05, projective=0.001)
        
        coords_h = h @ coords

        coords = geometry.from_homog_coords(coords.transpose(0, 1))
        coords_h = geometry.from_homog_coords(coords_h.transpose(0, 1))

        #coords[:,0] += self.W / 2
        #coords[:,1] += self.H / 2
        #coords_h[:,0] += self.W / 2
        #coords_h[:,1] += self.H / 2

        #coords[:,0] = (coords[:,0]/2.0)+0.5
        #coords[:,1] = (coords[:,1]/2.0)+0.5
        #coords[:,0] = (coords[:,0]/2.0)+0.5
        #coords[:,1] = (coords[:,1]/2.0)+0.5

        inliers = torch.rand(self.N_points) < 0.90
        offset = ((torch.rand((2, self.N_points))-0.5)*20) * ~inliers.expand(2, -1)
        w_gt = inliers.to(torch.float64)

        coords_h += offset.transpose(0,1)



        #coords[:,0] = (coords[:,0] / self.W) - 0.5
        #coords[:,1] = (coords[:,1] / self.H) - 0.5
        #coords_h[:,0] = (coords_h[:,0] / self.W) - 0.5
        #coords_h[:,1] = (coords_h[:,1] / self.H) - 0.5
        
        data = { 
            "p": coords,
            "ph": coords_h,
            "homography": h,
            "w_gt": w_gt
        }
        
        return data