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
from trainer import SfMTrainer
from debugger import Debugger as SfMDebugger
from point_trainer import PointTrainer
import tester

def parse_args(extra=[]):
    always = ["net", "workers", "device", "load"]
    return options.get_args(
        description="Train, debug or test a network",
        options=always + extra)

def main():

    choice = sys.argv.pop(1)

    if choice == "train":
        action = SfMTrainer(parse_args(["name", "batch", "train", "loss", "dataset"]))
    elif choice == "debugger":
        action = SfMDebugger(parse_args(["loss", "dataset"]))
    elif choice == "point":
        action = PointTrainer(parse_args())
    else:
        print("No such action to perform: %s" % choice)
        exit()

    action.run()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    main()