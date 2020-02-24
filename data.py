
import random
import numpy as np
import torch
from kitti import Kitti
from lyft import Lyft
from simple_dataset import SimpleDataset

kitti_path = "../Data/kitti2"
coco_path = "../Data/coco"
lyft_path = "/home/ai/Code/Data/lyft/v1.02-train/"

def _split_dataset(dataset, train, val, test):
    assert sum([train, val, test]) == 1.0
    split = np.floor(len(dataset) * np.array([train, val, test]))
    split = [int(s) for s in split]
    split[0] += len(dataset) - sum(split)
    return torch.utils.data.random_split(dataset, split)

def _data_loaders(dataset, train, val, test, batch_size, workers):
    train_set, val_set, test_set = _split_dataset(dataset, train, val, test)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)

    print(f"Dataset: {len(dataset)}")
    print(f"Train: {len(train_set)} / {len(train_loader) * batch_size}")
    print(f"Val: {len(val_set)} / {len(val_loader) * batch_size}")
    print(f"Test: {len(test_set)} / {len(test_loader) * batch_size}")
    
    return { "train": train_loader, "val": val_loader, "test": test_loader }

def _get_batch_loader_split(Class, path, batch, workers):
    random.seed(1337)
    dataset = Class(path)
    return _data_loaders(dataset, 0.899, 0.001, 0.1, batch, workers)

def _get_loader(Class, path, workers):
    dataset = Class(path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True)
    return loader

def get_batch_loader_split(args):
    if args.dataset == "kitti":
        return _get_batch_loader_split(Kitti, kitti_path, args.batch, args.workers)
    elif args.dataset == "lyft":
        return _get_batch_loader_split(Lyft, lyft_path, args.batch, args.workers)
    else:
        print("No dataset named: %s" % args.dataset)
        exit()

def get_loader(args):
    if args.dataset == "kitti":
        return _get_loader(Kitti, kitti_path, args.workers)
    elif args.dataset == "lyft":
        return _get_loader(Lyft, lyft_path, args.workers)
    else:
        print("No dataset named: %s" % args.dataset)
        exit()

def get_coco_batch_loader_split(args):
    return _get_batch_loader_split(SimpleDataset, coco_path, args.workers)