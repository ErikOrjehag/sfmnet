
import torch
import torch.nn as nn
from networks.sfmlearner_depth import SFMLearnerDepth
from networks.sfmlearner_pose import SFMLearnerPose
from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from networks.pose_decoder import PoseDecoder

def get_net(args):
  nets = { "sfmlearner": SFMLearner, "monodepth2": Monodepth2 }
  return nets[args.net]()

def stack_tgt_refs(tgt, refs):
  # Stack tgt and refs along color channel
  B, _, H, W = tgt.shape
  return torch.cat( ( tgt, refs.view(B, -1, H, W) ), axis=1)

class SFMLearner(nn.Module):

  def __init__(self):
    super().__init__()
    self.depth_net = SFMLearnerDepth()
    self.pose_net = SFMLearnerPose()

  def forward(self, inputs):
    # Do something with inputs
    depth_out = self.depth_net(inputs["tgt"])
    pose_out = self.pose_net(stack_tgt_refs(inputs["tgt"], inputs["refs"]))
    return { **depth_out, **pose_out }

class Monodepth2(nn.Module):

  def __init__(self):
    super().__init__()
    self.depth_encoder = ResnetEncoder(
      num_layers=18, 
      pretrained=False, 
      num_input_images=1)

    self.depth_decoder = DepthDecoder(
      num_ch_enc=self.depth_encoder.num_ch_enc, 
      scales=range(4), 
      num_output_channels=1, 
      use_skips=True)

    self.pose_encoder = ResnetEncoder(
      num_layers=18, 
      pretrained=False, 
      num_input_images=3)

    self.pose_decoder = PoseDecoder(
      num_ch_enc=self.pose_encoder.num_ch_enc,
      num_input_features=1,
      num_frames_to_predict_for=2)

  def forward(self, inputs):
    depth_out = self.depth_decoder(self.depth_encoder(inputs["tgt"]))    
    pose_out = self.pose_decoder( [ self.pose_encoder(stack_tgt_refs(inputs["tgt"], inputs["refs"])) ] )
    return { **depth_out, **pose_out }
