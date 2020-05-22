import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil
import colorsys

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

def draw_matches(img1, img2, pts1, pts2):
  H, W = img1.shape[:2]
  N = pts1.shape[0]
  img = np.concatenate((img1, img2), axis=1)
  i = 0
  for p1, p2 in zip(pts1, pts2):
    p2 += np.array([W, 0])
    img = cv2.line(img, tuple(p1), tuple(p2), pretty_color(i, N), 1)
    i += 1
  return img

def pretty_color(i, n):
  rgb = colorsys.hsv_to_rgb(((i*2)/(n-1)) % n, 0.8, 1.0)
  return (rgb[2]*255, rgb[1]*255, rgb[0]*255)