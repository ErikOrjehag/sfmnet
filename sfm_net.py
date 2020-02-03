
import torch.nn as nn
from depth_net import DepthNet
from pose_net import PoseExpNet

class SfmNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_net = DepthNet()
        self.pose_net = PoseExpNet()
        self.depth_net.init_weights()
        self.pose_net.init_weights()

    def forward(self, inputs):
        tgt, refs = inputs[:2]
        depth_out = self.depth_net(tgt)
        pose_out = self.pose_net(tgt, refs)
        return depth_out, pose_out