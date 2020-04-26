import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

def tensor2depthimg(depth):
  if len(depth.shape) == 3:
    depth = depth.squeeze(0)
  depth = depth.cpu().detach().numpy()
  vmax = np.percentile(depth, 95)
  vmin = depth.min()
  #depth = np.uint8(np.clip((depth-vmin) / (vmax - vmin), 0, 1) * 255)
  depth = np.uint8(np.clip(depth / vmax, 0, 1) * 255)
  #return cv2.applyColorMap(depth, cv2.COLORMAP_COOL)
  #return cv2.applyColorMap(depth, cv2.COLORMAP_RAINBOW)
  return cv2.applyColorMap(255-depth, cv2.COLORMAP_HOT)
  #vmax = np.percentile(depth, 95)
  #normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
  #mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
  #colored = (mapper.to_rgba(depth)[:,:,:3] * 255).astype(np.uint8)
  #return colored
def tensor2img(img):
  img = img.cpu().detach().numpy()
  return np.uint8(np.transpose(img, (1, 2, 0))[:,:,::-1] * 255)

def tensor2diffimg(img):
  img = img.cpu().detach().numpy()
  return np.uint8(np.transpose(img, (1, 2, 0))[:,:,::-1] * 255)