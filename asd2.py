import multiprocessing

import time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils
from sequence_dataset import SequenceDataset

def main():
    dataset = SequenceDataset("/home/ai/Data/kitti_formatted")
    loader, _, _ = utils.data_loaders(dataset, 1.0, 0.0, 0.0, 4, 1)

    writer = SummaryWriter()
    config_3d = { "camera": { "cls": "PerspectiveCamera", "fov": 75, "near": 0.1, "far": 5000.0 } }
    
    for i, batch in enumerate(loader):
        tgt, refs, K, Kinv = batch
        
        s = tgt.shape[-1]*tgt.shape[-2]
        verts = np.random.random((1, s, 3)) * 1.0
        colors = np.repeat(np.array([[[255, 0, 0]]]), s, 1)
        
        grid = torchvision.utils.make_grid(tgt, nrow=2)

        loss = 1/(0.01*i+1)
        
        writer.add_image("tgt", img_tensor=grid, global_step=i)
        writer.add_mesh("cloud", vertices=verts, colors=colors, config_dict=config_3d, global_step=i)
        writer.add_scalar("loss", scalar_value=loss, global_step=i)
        
        time.sleep(3.0)

    writer.close()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    main()