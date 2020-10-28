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

def tensor2flatimg(img):
  img = img.cpu().detach().numpy()
  return cv2.applyColorMap(np.uint8(255*img), cv2.COLORMAP_BONE)

def tensor2idximg(img, colors):
  img = img.cpu().detach().numpy()
  H, W = img.shape
  img2 = np.zeros((H,W,3), dtype=np.uint8)
  for i, color in enumerate(colors):
    img2[img==i] = color
  return img2

def tensor2maskimg(img):
  img = img.cpu().detach().numpy()
  H, W = img.shape
  img2 = np.zeros((H,W), dtype=np.uint8)
  img2[img==True] = 255
  return img2

def tensor2img(img):
  img = img.cpu().detach().numpy()
  return np.uint8(np.transpose(img, (1, 2, 0))[:,:,::-1] * 255)

def tensor2diffimg(img):
  img = img.cpu().detach().numpy()
  return np.uint8(np.transpose(img, (1, 2, 0))[:,:,::-1] * 255)

def draw_text(text, img):
  return cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

def draw_matches(img1, img2, pts1, pts2, inliers=None, draw_outliers=True):
  H, W = img1.shape[:2]
  N = pts1.shape[0]
  if inliers is None:
    inliers = np.array([True] * N)
  outliers = ~inliers
  img = np.concatenate((img1, img2), axis=1)

  def _draw_match(img, p1, p2, c, w=1):
    p1 = tuple(p1)
    p2 = tuple(p2)
    img = cv2.line(img, p1, p2, c, w)
    img = cv2.circle(img, p1, 4, c, w)
    img = cv2.circle(img, p2, 4, c, w)
    return img

  for i, p1p2 in enumerate(zip(pts1[inliers], pts2[inliers])):
    p1, p2 = p1p2
    p2 += np.array([W, 0])
    c = pretty_color(i, N)
    img = _draw_match(img, p1, p2, c)

  if draw_outliers:
    for i, p1p2 in enumerate(zip(pts1[outliers], pts2[outliers])):
      p1, p2 = p1p2
      p2 += np.array([W, 0])
      c = (0,0,255)
      img = _draw_match(img, p1, p2, c, w=2)

  try:
    return img.get()
  except:
    return img



def pretty_color(i, n):
  n = max(n,1)
  rgb = colorsys.hsv_to_rgb(((i*10)/(n-1)) % n, 0.8, 0.8)
  return (rgb[2]*255, rgb[1]*255, rgb[0]*255)