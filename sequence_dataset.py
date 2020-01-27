import torch.utils.data as data
import torchvision
import numpy as np
from imageio import imread
import cv2
import glob
import os
import sys

class SequenceDataset(data.Dataset):
  def __init__(self, root):
    super().__init__()
    self.sequences = sorted(glob.glob(os.path.join(root, "*", "")))
    self.intrinsics = [
      np.genfromtxt(os.path.join(sequence, "cam.txt"), 
        delimiter=",", dtype=np.float32).reshape((3, 3))
      for sequence in self.sequences]
    self.inv_intrinsics = [np.linalg.inv(intrinsics) for intrinsics in self.intrinsics]
    self.images = [
      sorted(glob.glob(os.path.join(sequence, "*.jpg"))) 
      for sequence in self.sequences]
    self.lens = [len(images) - 2 for images in self.images]
    self.len = sum(self.lens)
    self.cumlen = np.hstack((0, np.cumsum(self.lens)))

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    for i in range(len(self.cumlen)):
      if index - self.cumlen[i] < 0:
        i -= 1
        tgt = 1 + index - self.cumlen[i]
        seq = self.images[i]
        paths = [seq[tgt], seq[tgt-1], seq[tgt+1]]
        imgs = np.transpose([np.array(imread(path)).astype(np.float32)/255 for path in paths], (0, 3, 1, 2))
        tgt = imgs[0]
        refs = imgs[1:]
        K = self.intrinsics[i]
        Kinv = self.inv_intrinsics[i]
        return (tgt, refs), (tgt, refs, K, Kinv)

def tensor_depthshow(name, depth):
  d = depth.cpu().detach().numpy()
  cv2.imshow(name, cv2.applyColorMap(np.uint8(d / d.max() * 255), cv2.COLORMAP_JET))

def tensor_imshow(name, img):
  im = img.cpu().detach().numpy()
  cv2.imshow(name, np.transpose(im, (1, 2, 0))[:,:,::-1])

if __name__ == '__main__':
  dataset = SequenceDataset("/home/ai/Data/kitti_formatted")
  for i in range(210, len(dataset)):
    (tgt, refs), _ = dataset[i]
    img = np.concatenate((refs[0], tgt, refs[1]), axis=1)
    tensor_imshow("img", img)
    cv2.waitKey(0)