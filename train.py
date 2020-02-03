import multiprocessing

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import utils
import argparse
from sfm_loss import SfmLoss
from sfm_net import SfmNet
import cv2
import time
import random
from pathlib import Path
import os

def is_interval(step, interval):
  return step % (interval+1) == interval

def main():
  # Parse arguments
  parser = argparse.ArgumentParser(description="Train and eval sfm nets.")
  parser.add_argument("--name", default="", type=str, help="The run name.")
  parser.add_argument("--batch", default=4, type=int, help="The batch size.")
  parser.add_argument("--workers", default=4, type=int, help="The number of worker threads.")
  parser.add_argument("--device", default="cuda", type=str, help="The device to run on cpu/cuda.")
  parser.add_argument("--epochs", default=24, type=int, help="Max number of epochs.")
  parser.add_argument("--load", default="", type=str, help="Load state file.")
  args = parser.parse_args()
  print("\nCurrent arguments -> ", args, "\n")

  if args.device == "cuda" and not torch.cuda.is_available:
    print("CUDA is not available!")
    exit()

  # Setup checkpoint directory
  checkpoint_dir = f"./checkpoints/{args.name}/"
  if args.name != "":
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

  # Tensorboard
  writer = SummaryWriter(log_dir=f"runs/{args.name}")

  # Construct datasets
  random.seed(1337)
  train_loader, val_loader, _ = utils.get_kitti_split(args.batch, args.workers)

  # The model
  model = SfmNet()
  
  # Optimizer and loss function
  #lr = 0.0002
  lr = 0.00001
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
  criterion = SfmLoss()

  # Intervals
  LOG_INTERVAL = 200

  # Load
  if args.load != "":
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch_start = checkpoint["epoch"]
  else:
    epoch_start = 0

  # Train
  model.train()
  model = model.to(args.device)
  
  for epoch in range(epoch_start, args.epochs):

    running_loss = 0.0
    epoch_ts = time.time()

    N_steps = len(train_loader)
    for step, inputs in enumerate(train_loader):

      # Send inputs to device
      inputs = [inp.to(args.device) for inp in inputs]

      # Forward pass and loss
      with torch.enable_grad():
        outputs = model(inputs)
        loss = criterion(inputs, outputs)[0]

      # Backward pass
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      # Log information
      if is_interval(step, LOG_INTERVAL):
        percent = 100 * step / N_steps
        avg_loss = running_loss / LOG_INTERVAL
        running_loss = 0.0
        t_sample = (time.time() - epoch_ts) / step / args.batch
        t_sample_ms = 1000*t_sample
        eta = t_sample * args.batch * (N_steps - step)
        samples = (epoch * N_steps + step) * args.batch
        print(f"Epoch {epoch+1}/{args.epochs} ({percent:3.0f}%, eta: {eta:4.0f}s) " +
          f"| {samples:5} samples | {t_sample_ms:.0f} ms/sample -> loss: {avg_loss:.3f}")
        
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

  writer.close()


if __name__ == "__main__":
  multiprocessing.set_start_method('spawn', True)
  main()