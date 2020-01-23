
import torch.nn as nn
from depth_net import DepthNet
from pose_net import PoseExpNet

class SfmNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_net = DepthNet()
        self.pose_net = PoseExpNet()

    def forward(self, x):
        depth_out = self.depth_net(x)
        pose_out = self.pose_net(x)
        return depth_out, pose_out