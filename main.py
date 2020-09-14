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


from sfm_trainer import SFMTrainer
from sfm_debugger import SFMDebugger
from sfm_tester import SFMTester

from point_trainer import PointTrainer
from point_debugger import PointDebugger

from homography_trainer import HomographyTrainer
from homography_debugger import HomographyDebugger

def parse_args(extra=[], overwrite={}):
    always = ["net", "workers", "device", "load"]
    return options.get_args(
        description="Train, debug or test a network",
        options=always + extra,
        overwrite=overwrite)

def main():

    choice = sys.argv.pop(1)

    # Depth
    if choice == "sfm-train":
        action = SFMTrainer(parse_args(["name", "batch", "train", "loss", "dataset"]))
    elif choice == "sfm-debug":
        action = SFMDebugger(parse_args(["loss", "dataset"]))
    elif choice == "sfm-test":
        action = SFMTester(parse_args(["dataset", "loss"], overwrite={"batch": 1}))

    # Unsuperpoint
    elif choice == "point-train":
        action = PointTrainer(parse_args(["name", "batch", "train", "dataset"]))
    elif choice == "point-debug":
        action = PointDebugger(parse_args(["loss", "batch", "dataset"]))

    # Homography consensus
    elif choice == "fcons-train":
        action = HomographyTrainer(parse_args(["name", "batch", "train", "dataset"]))
    elif choice == "fcons-debug":
        action = HomographyDebugger(parse_args(["loss", "batch", "dataset"]))

    # Ooops..
    else:
        print("No such action to perform: %s" % choice)
        exit()

    action.run()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    main()