import multiprocessing

import sys
sys.path.insert(0, "/home/ai/Code/sfmnet")

import torch
import torch.nn as nn
import utils

from kitti import Kitti

def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

def bb_conv(in_channels, out_channels, last_layer=False):
    if last_layer:
        last = []
    else:
        last = [nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(inplace=True)]
    return nn.Sequential(
            conv(in_channels, out_channels),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(inplace=True),
            conv(out_channels, out_channels),
            *last
        )

def rel_to_abs(P):
    B, _, H, W = P.shape
    cols = torch.arange(0, W).view(1, 1, W).expand(B, H, W)
    rows = torch.arange(0, H).view(1, H, 1).expand(B, H, W)
    P[:,0] += cols
    P[:,1] += rows
    P *= 8
    return P

class UnsuperPoint(nn.Module):

    def __init__(self):
        super().__init__()

        self.N = 300

        self.backbone = nn.Sequential(
            bb_conv(3, 32),
            nn.MaxPool2d(kernel_size=2),
            bb_conv(32, 64),
            nn.MaxPool2d(kernel_size=2),
            bb_conv(64, 128),
            nn.MaxPool2d(kernel_size=2),
            bb_conv(128, 256, last_layer=True),
        )

        self.score_decoder = nn.Sequential(
            conv(256, 256),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),
            conv(256, 1),
            nn.Sigmoid(),
        )

        self.position_decoder = nn.Sequential(
            conv(256, 256),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),
            conv(256, 2),
            nn.Sigmoid(),
        )

        self.descriptor_decoder = nn.Sequential(
            conv(256, 256),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(inplace=True),
            conv(256, 256),
        )

    def forward(self, image):
        B, _, H, W = image.shape
        image = utils.normalize_image(image)
        features = self.backbone(image)
        S = self.score_decoder(features)
        P = rel_to_abs(self.position_decoder(features))
        F = self.descriptor_decoder(features)

        Sflat = S.view(B, -1)
        Pflat = P.view(B, 2, -1)
        Fflat = F.view(B, 256, -1)

        Smax, ids = torch.topk(Sflat, k=self.N, dim=1, largest=True, sorted=False)
        Pmax = torch.stack([Pflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)
        Fmax = torch.stack([Fflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)

        outputs = {
            "S": Smax,
            "P": Pmax,
            "F": Fmax
        }

        return outputs

class SiameseUnsuperPoint(nn.Module):

    def __init__(self):
        super().__init__()
        self.unsuperpoint = UnsuperPoint()

    def forward(self, data):
        A = self.unsuperpoint(data["original"])
        B = self.unsuperpoint(data["transformed"])
        outputs = {
            "A": A,
            "B": B
        }
        return outputs
        
class UnsuperLoss():

    def __init__(self):
        self.CORR_DIST = 8

    def __call__(self, data):
        AP = data["A"]["P"]
        AS = data["A"]["S"]
        AF = data["A"]["F"]
        BP = data["B"]["P"]
        BS = data["B"]["S"]
        BF = data["B"]["F"]

        C = (AP.permute(0,2,1).unsqueeze(2) - BP.permute(0,2,1).unsqueeze(1)).norm(dim=-1)

        Cmin, ids = torch.min(C, dim=1)
        mask = Cmin.le(self.CORR_DIST)

        d = Cmin[mask]
        dmean = d.mean()

        loss_position = d.sum()
        
        #BS_ = torch.stack([BS[i,ids[i]] for i in range(ids.shape[0])], dim=0)
        #loss_score = ((AS - BS_)[mask] ** 2).sum()

        mask = mask.view(-1)
        ids = ids.view(-1)
        AS_ = AS.view(-1)[mask]
        BS_ = BS.view(-1)[ids][mask]

        loss_score = (AS_ - BS_) ** 2

        S_ = (AS_ + BS_) / 2
        loss_usp = S_ * (d - dmean)

        x = 1

        return 0, {}

def main():
    dataset = Kitti("/home/ai/Code/Data/kitti2")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False,
        num_workers=1, pin_memory=True)

    loss_fn = UnsuperLoss()

    model = SiameseUnsuperPoint()
    model.train()

    for data in loader:
        output = model.forward({ "original": data["tgt"], "transformed": data["tgt"] })
        loss, debug = loss_fn(output)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    main()