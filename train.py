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

def is_interval(step, interval):
  return step % (interval+1) == interval

def main():
  # Parse arguments
  args = options.get_args(
    description="Train a network",
    options=[
      "name", 
      "net",
      "batch", 
      "workers", 
      "device", 
      "lr", 
      "epochs", 
      "load", 
      "log-interval",
      "loss"
    ])

  # Construct datasets
  random.seed(1337)
  train_loader, val_loader, _ = utils.get_kitti_split(args.batch, args.workers)

  # The model architecture
  model = networks.architectures.get_net(args)

  # The loss
  loss_fn = sfm_loss.get_loss_fn(args)

  # Optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

  # Load
  if args.load != "":
    checkpoint = torch.load(args.load, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch_start = checkpoint["epoch"]
  else:
    epoch_start = 0

  # Setup checkpoint directory
  checkpoint_dir = f"./checkpoints/{args.name}/"
  if args.name != "":
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

  # Tensorboard
  if args.name != "":
    writer = SummaryWriter(log_dir=f"runs/{args.name}")

  # Train
  model.train()
  model = model.to(args.device)
  
  for epoch in range(epoch_start, args.epochs):

    running_loss = 0.0
    epoch_ts = time.time()

    N_steps = len(train_loader)
    for step, inputs in enumerate(train_loader):

      # Send inputs to device
      inputs = { k: v.to(args.device) for k, v in inputs.items() }

      # Forward pass and loss
      with torch.enable_grad():
        loss, data = utils.forward_pass(model, loss_fn, inputs)

      # Backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      # Log information
      if is_interval(step, args.log_interval):
        percent = 100 * step / N_steps
        avg_loss = running_loss / args.log_interval
        running_loss = 0.0
        t_sample = (time.time() - epoch_ts) / step / args.batch
        t_sample_ms = 1000*t_sample
        eta = t_sample * args.batch * (N_steps - step)
        samples = (epoch * N_steps + step) * args.batch
        print(f"Epoch {epoch+1}/{args.epochs} ({percent:3.0f}%, eta: {eta:4.0f}s) " +
          f"| {samples:5} samples | {t_sample_ms:.0f} ms/sample -> loss: {avg_loss:.3f}")
        
        if args.name != "":
          writer.add_scalar("loss", scalar_value=avg_loss, global_step=samples)
    
    # Checkpoint end of epoch
    if args.name != "":
      checkpoint_file = f"epoch_{epoch+1}.pt"
      path = os.path.join(checkpoint_dir, checkpoint_file)
      print(f"Save checkpoint: {path}")
      torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
      }, path)

  if args.name != "":
    writer.close()


if __name__ == "__main__":
  multiprocessing.set_start_method('spawn', True)
  main()