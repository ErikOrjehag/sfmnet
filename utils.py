import torch
import numpy as np

def split_dataset(dataset, train, val, test):
  assert sum([train, val, test]) == 1.0
  split = np.floor(len(dataset) * np.array([train, val, test]))
  split = [int(s) for s in split]
  split[0] += len(dataset) - sum(split)
  return torch.utils.data.random_split(dataset, split)

def data_loaders(dataset, train, val, test, batch_size, workers):
  train_set, val_set, test_set = split_dataset(dataset, train, val, test)
  print("Dataset: %d, Train: %d, Val: %d, Test: %d" % 
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