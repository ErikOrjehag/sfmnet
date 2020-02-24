import torch.utils.data as data
import torchvision
import numpy as np
import cv2
import glob
import os
import sys
import viz
import torch
from imageio import imread
import utils

def relative_transform(from_T, to_T):
  R1, t1 = from_T[:3,:3], from_T[:3,3:]
  R2, t2 = to_T[:3,:3], to_T[:3,3:]
  rel_T = np.vstack((np.hstack((R2.T @ R1, R2.T @ (t1 - t2))), (0,0,0,1)))
  return rel_T

class SequenceDataset(data.Dataset):

  def __init__(self, root, load_gt):
    super().__init__()

    self.root = root
    self.load_gt = load_gt

    self.imHW = (128, 416)

    # Sequences
    self.sequences = self._get_sequences()
    
    # Samples
    self.samples = [self._get_samples(sequence) for sequence in self.sequences]
    
    # Length helpers
    self.lens = [len(samples) - 2 for samples in self.samples]
    self.len = sum(self.lens)
    self.cumlen = np.hstack((0, np.cumsum(self.lens)))

  def _get_sequences(self):
    raise NotImplementedError("_get_sequences")

  def _get_samples(self, sequence):
    raise NotImplementedError("_get_samples")

  def _load_sample(self, sample):
    raise NotImplementedError("_load_sample")

  def _load_depth(self, sample):
    raise NotImplementedError("_load_depth")

  def calc_crop(self, H, W):
    imH, imW = self.imHW
    scale = imW / W
    crop = int((H * scale - imH) / 2)
    return crop, scale

  def _load_image_from_disk(self, path, K):
    img = imread(path)
    H, W = img.shape[:2]
    #print(img.shape)
    crop, scale = self.calc_crop(H, W)
    #print(crop, scale)
    #print(K)
    if scale != 1:
      img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
      #print(img.shape)
      img = img[crop:crop+self.imHW[0]]
      #print(img.shape)
      K[:2,:] *= scale
      #print(K)
      K[1,2] -= crop
      #print(K)
    img = utils.cv2_to_torch(img)
    return img, K

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    for i in range(len(self.cumlen)):
      if index - self.cumlen[i] < 0:
        i -= 1
        tgt = 1 + index - self.cumlen[i]
        sequence = self.samples[i]
        samples = [sequence[tgt], sequence[tgt-1], sequence[tgt+1]]
        data = [self._load_sample(i, sample) for sample in samples]
        
        tgt_img, tgt_T, K = data[0]
        refs_data = data[1:]
        refs_img = [d[0] for d in refs_data]
        refs_T = [d[1] for d in refs_data]

        sparse = self._load_depth(samples[0])

        inputs = {
          "tgt": torch.tensor(tgt_img),
          "refs": torch.tensor(refs_img),
          "K": torch.tensor(K),
          "gt_sparse": torch.tensor(sparse).unsqueeze(0)
        }

        return inputs
