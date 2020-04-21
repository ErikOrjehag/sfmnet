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
import histogram
import random

def relative_transform(from_T, to_T):
  R1, t1 = to_T[:3,:3], to_T[:3,3:]
  R2, t2 = from_T[:3,:3], from_T[:3,3:]
  rel_T = np.vstack((np.hstack((R2.T @ R1, R2.T @ (t1 - t2))), (0,0,0,1)))
  return rel_T

class SequenceDataset(data.Dataset):

  def __init__(self, root, split=None):
    super().__init__()

    self.root = root

    self.imHW = (128, 416)

    # Sequences
    sequences = self._get_sequences()

    TEST_RATIO = 0.1

    if split is None:
      self.sequences == sequences
    else:
      test_n = int(TEST_RATIO * len(sequences))
      random.seed(1337)
      random.shuffle(sequences)
      test_sequences = sequences[:test_n]
      train_sequences = sequences[test_n:]
      if split == "test":
        print("Test sequences: ", len(test_sequences))
        self.sequences = test_sequences
      elif split == "train":
        print("Train sequences: ", len(train_sequences))
        self.sequences = train_sequences
      else:
        print("INVALID SPLIT: ", split)
        exit()

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
    
    crop, scale = self.calc_crop(H, W)
    
    if scale != 1:
      img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
      img = img[crop:crop+self.imHW[0]]
      K[:2,:] *= scale
      K[1,2] -= crop
    
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
        #data[0][0] = (data[0][0]*0.5).astype(np.int)
        #for i in range(1, len(data)):
        #  data[i][0] = histogram.match_histograms(data[i][0], data[0][0])
        for i in range(len(data)):
          data[i][0] = utils.cv2_to_torch(data[i][0])
        
        tgt_img, tgt_T, K = data[0]
        refs_data = data[1:]
        refs_img = [d[0] for d in refs_data]
        refs_T = [d[1] for d in refs_data]

        T = [relative_transform(tgt_T, ref_T) for ref_T in refs_T]

        sparse = self._load_depth(samples[0])

        inputs = {
          "tgt": tgt_img,
          "refs": torch.stack(refs_img),
          "K": torch.tensor(K),
          "gt_sparse": torch.tensor(sparse).unsqueeze(0),
          "tgt_i": tgt,
          "T": torch.tensor(T),
          "tgt_T": torch.tensor(tgt_T)
        }

        return inputs
