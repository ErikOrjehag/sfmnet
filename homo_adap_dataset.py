
import os
import glob
import torch
import torch.utils.data as data
from imageio import imread
import utils
import homography
import numpy as np
import cv2

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
        #h = homography.random_homography(rotation=np.pi/8, translation=50, scale=0.5, sheer=0.1, projective=0.001)
        #h = homography.random_homography(rotation=0, translation=0, scale=0.2, sheer=0, projective=0)
        h = homography.random_homography(rotation=np.pi/20, translation=10, scale=0.2, sheer=0.05, projective=0.001)
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