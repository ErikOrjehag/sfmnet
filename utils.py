import torch
import numpy as np
from sequence_dataset import SequenceDataset
import cv2

def iterate_loader(args, loader, fn):
  for step, inputs in enumerate(loader):
    inputs = { k: v.to(args.device) for k, v in inputs.items() }
    fn(step, inputs)

def forward_pass(model, loss_fn, inputs):
  outputs = model(inputs)
  data = { **inputs, **outputs }
  loss, debug = loss_fn(data)
  data = { **data, **debug }
  return loss, data

def normalize_map(map):
  mean_map = map.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
  norm_map = map / (mean_map + 1e-7)
  return norm_map

def randn_like(tensor):
  return torch.randn(tensor.shape).to(tensor.type)

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