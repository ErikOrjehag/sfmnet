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
  dataset = SequenceDataset("../Data/kitti2")
  train_loader, val_loader, test_loader = data_loaders(
    dataset, 0.899, 0.001, 0.1, batch, workers)
  return train_loader, val_loader, test_loader