
import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import utils

class BaseTrainer():

    def __init__(self, args, loaders, model, loss_fn):
        super().__init__()
        
        self.BATCH = args.batch
        self.EPOCHS = args.epochs
        self.DEVICE = args.device
        self.LOG_INTERVAL = args.log_interval
        self.SHOULD_VALIDATE = args.validate

        # Train, Val, Test loaders
        self.loaders = loaders
        # print number of samples
        print("Train images: ", len(self.loaders["train"]))

        # The model to train
        self.model = model.to(self.DEVICE)

        # The loss function
        self.loss_fn = loss_fn

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))

        # Load
        self.EPOCH_START = 0
        self.load_checkpoint(args)

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

            utils.iterate_loader(self.DEVICE, self.loaders["train"], self.__train_step_fn)

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

    def __train_step_fn(self, step, inputs):

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
                val_metrics, train_metrics = self.__validate()

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

    def __validate(self):
        N = len(self.loaders["val"])
        val_metrics = {}
        train_metrics = {}
        
        utils.iterate_loader(self.DEVICE, self.loaders["val"], self.__val_step_fn, args=(val_metrics,))
        utils.iterate_loader(self.DEVICE, self.loaders["train"], self.__val_step_fn, args=(train_metrics,), end=N)

        val_metrics = utils.map_dict(val_metrics, lambda v: v/N)
        train_metrics = utils.map_dict(train_metrics, lambda v: v/N)

        return val_metrics, train_metrics

    def __val_step_fn(self, step, inputs, metrics_sum):
        with torch.no_grad():
            loss, data = utils.forward_pass(self.model, self.loss_fn, inputs)
        metrics = self.calc_metrics(data)
        utils.sum_to_dict(metrics_sum, metrics)

    # This can be overridden in child classes
    def calc_metrics(self, data):
        return {}

    # This can be overridden in child classes
    def load_checkpoint(self, args):
        checkpoint = torch.load(args.load, map_location=torch.device(args.device))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.EPOCH_START = checkpoint["epoch"]