import multiprocessing

import wandb
import time
import utils
from sequence_dataset import SequenceDataset
import numpy as np

def main():
    dataset = SequenceDataset("/home/ai/Data/kitti_formatted")
    loader, _, _ = utils.data_loaders(dataset, 1.0, 0.0, 0.0, 4, 1)

    wandb.init(project="sfmnet")

    for i, batch in enumerate(loader):
        tgt, refs, K, Kinv = batch
        #wandb.log({ "batch": wandb.Video((tgt*255).byte(), fps=1, format="webm") }, step=i)
        pos = np.random.random((tgt.shape[-1]*tgt.shape[-2], 3)) * 100.0
        color = np.random.randint(0, 256, (tgt.shape[-1]*tgt.shape[-2],3))
        cloud = np.concatenate((pos, color), axis=1)
        print(cloud.shape)
        wandb.log({ 
            "batch": [wandb.Image(img) for img in tgt],
            "loss": 1/(0.01*i+1),
            "cloud": wandb.Object3D(cloud)
            }, step=i)
        time.sleep(3.0)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    main()