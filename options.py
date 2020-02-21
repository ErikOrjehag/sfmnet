
import argparse
import torch

def get_args(description, options):
  parser = argparse.ArgumentParser(description=description)
  if "name" in options:
    parser.add_argument("--name", default="", type=str, help="The run name.")
  if "batch" in options:
    parser.add_argument("--batch", default=4, type=int, help="The batch size.")
  if "workers" in options:
    parser.add_argument("--workers", default=4, type=int, help="The number of worker threads.")
  if "device" in options:
    parser.add_argument("--device", default="cuda", type=str, help="The device to run on cpu/cuda.")
  if "dataset" in options:
    parser.add_argument("--dataset", default="kitti", type=str, choices=["kitti", "lyft"], help="Which dataset (kitti/lyft).")
  
  if "load" in options:
    parser.add_argument("--load", default="", type=str, help="Load state file.")
  
  if "train" in options:
    parser.add_argument("--lr", default=0.0002, type=float, help="Learning rate.")
    parser.add_argument("--epochs", default=30, type=int, help="Max number of epochs.")
    parser.add_argument("--log-interval", default=500, type=int, help="Number of samples before logging average running loss.")
    parser.add_argument("--validate", default=False, action="store_true", help="Validate model against validation dataset every logging interval.")
  
  if "loss" in options:
    parser.add_argument("--smooth-weight", default=0.1, type=float, help="Smooth loss weight.")
    parser.add_argument("--explain-weight", default=0.2, type=float, help="Explainability mask reqularization loss weight.")
    parser.add_argument("--ssim-weight", default=0.0, type=float, help="SSIM/L1 loss balancing weight.")
    parser.add_argument("--which-smooth-map", default="depth", type=str, choices=["depth", "disp"], help="Use depth or disp in smooth loss.")
    parser.add_argument("--smooth-map-normalization", default=False, action="store_true", help="Normalize depth/disp before smoothness loss.")
    parser.add_argument("--edge-aware", default=False, action="store_true", help="Use edge aware smoothness loss.")
    parser.add_argument("--upscale", default=False, action="store_true", help="Upscale intermediate depth maps to original image size.")
    parser.add_argument("--stationary-mask", default=False, action="store_true", help="Remove stationary pixels from the loss.")
    parser.add_argument("--min-loss", default=False, action="store_true", help="Use minimum instead of average loss across referance frames.")
  
  if "net" in options:
    parser.add_argument("--net", default="monodepth2", type=str, choices=["sfmlearner", "monodepth2"], help="The model architecture to use.")
    parser.add_argument("--min-depth", default=0.1, type=float, help="Minimum depth predicted.")
    parser.add_argument("--max-depth", default=100, type=float, help="Maximum depth predicted.")
  
  args = parser.parse_args()
  print("\nCurrent arguments -> ", args, "\n")

  if args.device == "cuda" and not torch.cuda.is_available():
    print("CUDA is not available!")
    exit()

  return args