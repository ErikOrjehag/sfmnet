
import random
import numpy as np
import torch
from kitti import Kitti
from lyft import Lyft
from homo_adap_dataset import HomoAdapDataset, HomoAdapDatasetCocoKittiLyft, HomoAdapDatasetFromSequences
import itertools

kitti_path = "/home/ai/Code/Data/kitti2"
coco_path = "/home/ai/Code/Data/coco/unlabeled2017/"
lyft_path = "/home/ai/Code/Data/lyft/v1.02-train/"
lyft_kittistyle_path = "/home/ai/Code/Data/lyft2"

def _split_dataset(dataset, train, val, test):
    assert sum([train, val, test]) == 1.0
    split = np.floor(len(dataset) * np.array([train, val, test]))
    split = [int(s) for s in split]
    split[0] += len(dataset) - sum(split)
    random.seed(1337)
    return torch.utils.data.random_split(dataset, split)
    #print(split)
    #train = itertools.islice(dataset, 0, split[0])
    #val = itertools.islice(dataset, split[0], split[0]+split[1])
    #test = itertools.islice(dataset, split[0]+split[1], split[0]+split[1]+split[2])
    #dataset[0:split[0]]
    #dataset[split[0]:split[0]+split[1]]
    #dataset[split[0]+split[1]:split[0]+split[1]+split[2]]
    #return train, val, test

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
    dataset = Class(path)
    return _data_loaders(dataset, 0.9, 0.0, 0.1, batch, workers)

def _get_batch_sequence_loader_split(Class, path, batch, workers):
    train_set = Class(path, "train")
    test_set = Class(path, "test")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)
    return { "train": train_loader, "val": None, "test": test_loader }

def _get_loader(Class, path, workers):
    dataset = Class(path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True)
    return loader

def get_batch_loader_split(args):
    if args.dataset == "kitti":
        return _get_batch_sequence_loader_split(Kitti, kitti_path, args.batch, args.workers)
    elif args.dataset == "lyft":
        return _get_batch_sequence_loader_split(Lyft, lyft_path, args.batch, args.workers)
    elif args.dataset == "lyft_kittistyle":
        return _get_batch_sequence_loader_split(Kitti, lyft_kittistyle_path, args.batch, args.workers)
    elif args.dataset == "coco_homo_adapt":
        return _get_batch_loader_split(HomoAdapDataset, coco_path, args.batch, args.workers)
    elif args.dataset == "cocokittylyft_homo_adapt":
        return _get_batch_loader_split(HomoAdapDatasetCocoKittiLyft, coco_path, args.batch, args.workers)
    elif args.dataset == "kitti_homo_adapt":
        return _get_batch_loader_split(HomoAdapDatasetFromSequences, kitti_path, args.batch, args.workers)
    elif args.dataset == "lyft_homo_adapt":
        return _get_batch_loader_split(HomoAdapDatasetFromSequences, lyft_path, args.batch, args.workers)
    else:
        print("No dataset named: %s" % args.dataset)
        exit()

def get_loader(args):
    if args.dataset == "kitti":
        return _get_loader(Kitti, kitti_path, args.workers)
    elif args.dataset == "lyft":
        return _get_loader(Lyft, lyft_path, args.workers)
    elif args.dataset == "lyft_kittistyle":
        return _get_loader(Kitti, lyft_kittistyle_path, args.workers)
    elif args.dataset == "coco_homo_adapt":
        return _get_loader(HomoAdapDataset, coco_path, args.workers)
    elif args.dataset == "cocokittylyft_homo_adapt":
        return _get_loader(HomoAdapDatasetCocoKittiLyft, coco_path, args.workers)
    elif args.dataset == "kitti_homo_adapt":
        return _get_loader(HomoAdapDatasetFromSequences, kitti_path, args.workers)
    elif args.dataset == "lyft_homo_adapt":
        return _get_loader(HomoAdapDatasetFromSequences, lyft_path, args.workers)
    else:
        print("No dataset named: %s" % args.dataset)
        exit()