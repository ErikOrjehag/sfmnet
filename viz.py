import numpy as np
import cv2

def tensor2depthimg(depth):
  depth = depth.cpu().detach().numpy()
  depth = np.uint8(depth / depth.max() * 255)
  return cv2.applyColorMap(depth, cv2.COLORMAP_COOL)

def tensor2img(img):
  img = img.cpu().detach().numpy()
  return np.uint8((np.transpose(img, (1, 2, 0))[:,:,::-1] + 1) / 2 * 255)

def tensor2diffimg(img):
  img = img.cpu().detach().numpy()
  return np.uint8(np.transpose(img, (1, 2, 0))[:,:,::-1] * 255)