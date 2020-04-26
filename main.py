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
from sfm_trainer import SfMTrainer
from debugger import Debugger as SfMDebugger
from sfm_tester import SfMTester
from point_trainer import PointTrainer
from debugger_point import DebuggerPoint

def parse_args(extra=[], overwrite={}):
    always = ["net", "workers", "device", "load"]
    return options.get_args(
        description="Train, debug or test a network",
        options=always + extra,
        overwrite=overwrite)

def main():

    choice = sys.argv.pop(1)

    if choice == "sfm-train":
        action = SfMTrainer(parse_args(["name", "batch", "train", "loss", "dataset"]))
    elif choice == "sfm-debug":
        action = SfMDebugger(parse_args(["loss", "dataset"]))
    elif choice == "sfm-test":
        action = SfMTester(parse_args(["dataset", "loss"], overwrite={"batch": 1}))
    elif choice == "point-train":
        action = PointTrainer(parse_args(["name", "batch", "train"]))
    elif choice == "point-debug":
        action = DebuggerPoint(parse_args(["loss"]))
    else:
        print("No such action to perform: %s" % choice)
        exit()

    action.run()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    main()