import multiprocessing

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import utils
import argparse
import sfm_loss
import networks.architectures
import cv2
import time
import random
from pathlib import Path
import os
import options
import sys
from trainer import Trainer
from debugger import Debugger
import tester

def parse_args(extra=[]):
    always = ["net", "workers", "device", "load", "loss", "dataset"]
    return options.get_args(
        description="Train, debug or test a network",
        options=always + extra)

def main():

    choice = sys.argv.pop(1)

    if choice == "train":
        action = Trainer(parse_args(["name", "batch", "train"]))
    elif choice == "debugger":
        action = Debugger(parse_args())
    else:
        print("No action to perform: (choose train/debug/test)")
        exit()

    action.run()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    main()