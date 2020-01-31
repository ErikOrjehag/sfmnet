import torch
import numpy as np
from sequence_dataset import SequenceDataset
import cv2

def split_dataset(dataset, train, val, test):
  assert sum([train, val, test]) == 1.0
  split = np.floor(len(dataset) * np.array([train, val, test]))
  split = [int(s) for s in split]
  split[0] += len(dataset) - sum(split)
  return torch.utils.data.random_split(dataset, split)

def data_loaders(dataset, train, val, test, batch_size, workers):
  train_set, val_set, test_set = split_dataset(dataset, train, val, test)
  print("Dataset: %d, Train: %d, Val: %d, Test: %d\n" % 
    (len(dataset), len(train_set), len(val_set), len(test_set)))
  train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)
  val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)
  return train_loader, val_loader, test_loader

def get_kitti_split(batch, workers):
  dataset = SequenceDataset("../Data/kitti")
  train_loader, val_loader, test_loader = data_loaders(
    dataset, 0.799, 0.001, 0.2, batch, workers)
  return train_loader, val_loader, test_loader

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