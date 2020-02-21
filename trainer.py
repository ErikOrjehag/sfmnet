import multiprocessing

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import utils
import data
import argparse
import sfm_loss
import networks.architectures
import cv2
import time
from pathlib import Path
import os
import options
import evaluation

class Trainer():

    def __init__(self, args):
        super().__init__()
        
        self.BATCH = args.batch
        self.EPOCHS = args.epochs
        self.DEVICE = args.device
        self.LOG_INTERVAL = args.log_interval
        self.SHOULD_VALIDATE = args.validate

        # Construct datasets
        self.loaders = data.get_batch_loader_split(args)
        
        # The model architecture
        self.model = networks.architectures.get_net(args)

        # The loss
        self.loss_fn = sfm_loss.get_loss_fn(args)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

        # Load
        if args.load != "":
            checkpoint = torch.load(args.load, map_location=torch.device(args.device))
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.EPOCH_START = checkpoint["epoch"]
        else:
            self.EPOCH_START = 0

        # Setup checkpoint directory
        self.SHOULD_CHECKPOINT = (args.name != "")
        if self.SHOULD_CHECKPOINT:
            self.checkpoint_dir = f"./checkpoints/{args.name}/"
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Tensorboard
        self.SHOULD_WRITE = (args.name != "")
        if self.SHOULD_WRITE:
            self.writer = SummaryWriter(log_dir=f"runs/{args.name}")

    def run(self):
        
        self.model.train()

        for epoch in range(self.EPOCH_START, self.EPOCHS):

            self.epoch = epoch
            self.epoch_ts = time.time()
            self.running_loss = 0.0

            utils.iterate_loader(self.DEVICE, self.loaders["train"], self._train_step_fn)

            if self.SHOULD_CHECKPOINT:
                checkpoint_file = f"epoch_{epoch+1}.pt"
                path = os.path.join(self.checkpoint_dir, checkpoint_file)
                print(f"Saving checkpoint: {path}")
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }, path)

        if self.SHOULD_WRITE:
            self.writer.close()

    def _train_step_fn(self, step, inputs):

        # Forward pass and loss
        with torch.enable_grad():
            loss, data = utils.forward_pass(self.model, self.loss_fn, inputs)

        # Backward pass
        utils.backward_pass(self.optimizer, loss)
        
        # Save loss
        self.running_loss += loss.item()

        # Log information
        if utils.is_interval(step, self.LOG_INTERVAL):

            if self.SHOULD_VALIDATE:
                val_metrics, train_metrics = self._validate()

            N_steps = len(self.loaders["train"])
            percent = 100 * step / N_steps
            avg_loss = self.running_loss / self.LOG_INTERVAL
            self.running_loss = 0.0
            t_sample = (time.time() - self.epoch_ts) / step / self.BATCH
            t_sample_ms = 1000*t_sample
            eta = t_sample * self.BATCH * (N_steps - step)
            samples = (self.epoch * N_steps + step) * self.BATCH
            print(f"Epoch {self.epoch+1}/{self.EPOCHS} ({percent:3.0f}%, eta: {utils.sec_to_hms(eta)}) " +
            f"| {samples:5} samples | {t_sample_ms:.0f} ms/sample -> loss: {avg_loss:.3f}")
            
            if self.SHOULD_WRITE:
                self.writer.add_scalar("loss", scalar_value=avg_loss, global_step=samples)
                if self.SHOULD_VALIDATE:
                    for key in val_metrics.keys():
                        if key == "abs_rel":
                            self.writer.add_scalar(f"val/{key}", scalar_value=val_metrics[key], global_step=samples)
                            self.writer.add_scalar(f"train/{key}", scalar_value=train_metrics[key], global_step=samples)
    
    def _validate(self):
        N = len(self.loaders["val"])
        val_metrics = {}
        train_metrics = {}
        
        utils.iterate_loader(self.DEVICE, self.loaders["val"], self._val_step_fn, args=(val_metrics,))
        utils.iterate_loader(self.DEVICE, self.loaders["train"], self._val_step_fn, args=(train_metrics,), end=N)

        val_metrics = utils.map_dict(val_metrics, lambda v: v/N)
        train_metrics = utils.map_dict(train_metrics, lambda v: v/N)

        return val_metrics, train_metrics
        
    def _val_step_fn(self, step, inputs, metrics_sum):
        with torch.no_grad():
            loss, data = utils.forward_pass(self.model, self.loss_fn, inputs)
        gt_depth = data["gt_sparse"]
        pred_depth = data["depth"][0]
        metrics = evaluation.eval_depth(gt_depth, pred_depth)
        utils.sum_to_dict(metrics_sum, metrics)