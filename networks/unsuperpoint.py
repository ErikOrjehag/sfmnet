import multiprocessing

import sys
sys.path.insert(0, "/home/ai/Code/sfmnet")

import torch
import torch.nn as nn
import utils

from geometry import from_homog_coords, to_homog_coords

from homo_adap_dataset import HomoAdapDataset

import reconstruction

def conv(in_channels, out_channels):
    # Create conv2d layer
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

def bb_conv(in_channels, out_channels, last_layer=False):
    # Create 2 conv layers separated by batch norm and leaky relu
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
    # Convert local (relative) positions P to global pixel positions
    B, _, H, W = P.shape
    cols = torch.arange(0, W, device=P.device).view(1, 1, W).expand(B, H, W)
    rows = torch.arange(0, H, device=P.device).view(1, H, 1).expand(B, H, W)
    return (P + torch.stack((cols, rows), dim=1)) * 8

def decorrelate(F):
    # Create a correlation matrix of feature vector F than can be
    # used to formulate a decorrelation loss
    f = F.permute(0,2,1)
    mean = f.mean(dim=-1, keepdims=True)
    b = f - mean
    dot = (b.unsqueeze(2) * b.unsqueeze(1)).sum(dim=-1)
    d = torch.sqrt(dot.diagonal(dim1=1,dim2=2))
    dd = d.unsqueeze(2) * d.unsqueeze(1)
    R = dot / dd
    idx = torch.arange(0,R.shape[1],out=torch.LongTensor())
    R[:,idx,idx] = 0
    return R**2

def uniform_distribution_loss(values, a=0., b=1.):
    # Create a loss that enforces uniform distribution 
    # of values in the interval [a, b]
    v = torch.sort(values.flatten())[0]
    L = v.shape[0]
    i = torch.arange(1, L+1, dtype=torch.float).to(values.device)
    s = ( (v-a) / (b-a) - (i-1) / (L-1) )**2
    return s

def brute_force_match(AF, BF):
    # Brute force match descriptor vectors [B,256,N]
    af = AF.permute(0,2,1).unsqueeze(2) # [B,N,256]
    bf = BF.permute(0,2,1).unsqueeze(1)
    l2 = (af - bf).norm(dim=-1) # [B,N,N]
    Aids = torch.argmin(l2, dim=1) # [B,N]
    Bids = torch.argmin(l2, dim=2) # [B,N]
    B, N = Bids.shape
    #offset = (torch.arange(0,B)*N).to(AF.device).repeat_interleave(N)
    #match = Aids.flatten()[offset+Bids.flatten()].reshape(B,N) # [B,N]
    match = torch.stack([ Aids[b][Bids[b]] for b in range(B) ])
    asc = torch.arange(0, N).repeat(B,1).to(match.device)
    crossCheckMask = (match == asc)
    return Bids, crossCheckMask
    
class UnsuperPoint(nn.Module):

    def __init__(self, N):
        super().__init__()

        self.N = N

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
        
        # CNN (joint backbone, separate decoder heads)
        features = self.backbone(image)
        S = self.score_decoder(features)
        Prel = self.position_decoder(features)
        F = self.descriptor_decoder(features) # [B, C, H/8, W/8]
        
        # Relative to absolute pixel coordinates
        P = rel_to_abs(Prel) # [B, 2, H/2, W/2]

        # Interpolate feature descriptors
        sampling_grid = reconstruction.to_sampling_grid(P, HW=(H,W)) # [B, H/2, W/2, 2] -1 to 1
        F_interpolated = nn.functional.grid_sample(F, sampling_grid, padding_mode="border", align_corners=True)
        
        # Flatten
        Sflat = S.view(B, -1)
        Pflat = P.view(B, 2, -1)
        Prelflat = Prel.view(B, 2, -1)
        Fflat = F_interpolated.view(B, 256, -1)

        N_good_score = torch.min((Sflat > 0.2).sum(dim=1))
        #print("N_good_score", N_good_score)

        # Get data with top N score (S) 450
        Smax, ids = torch.topk(Sflat, k=350, dim=1, largest=True, sorted=False)
        Pmax = torch.stack([Pflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)
        #Prelmax = torch.stack([Prelflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)
        Fmax = torch.stack([Fflat[i,:,ids[i]] for i in range(ids.shape[0])], dim=0)

        outputs = {
            "S": Sflat,
            "P": Pflat,
            "F": Fflat,
            "Prel": Prelflat,
            "Smax": Smax,
            "Pmax": Pmax,
            "Fmax": Fmax,
        }

        return outputs

class SiameseUnsuperPoint(nn.Module):

    def __init__(self, N=300):
        super().__init__()
        self.unsuperpoint = UnsuperPoint(N=N)

    def forward(self, data):
        if "img" in data and "warp" in data:
            imgA = data["img"]
            imgB = data["warp"]
        elif "tgt" in data and "refs" in data:
            imgA = data["tgt"]
            imgB = data["refs"][:,1]
            data["img"] = imgA
            data["warp"] = imgB
            print(imgA.shape)
            print(imgB.shape)
        else:
            print("INVALID INPUT DATA")
            exit()
        A = self.unsuperpoint(imgA)
        B = self.unsuperpoint(imgB)
        outputs = {
            "A": A,
            "B": B
        }
        return outputs

class SequenceUnsuperPoint(nn.Module):

    def __init__(self, N=300):
        super().__init__()
        self.unsuperpoint = UnsuperPoint(N=N)

    def forward(self, data):
        points_tgt = self.unsuperpoint(data["tgt"])
        points_refs = []
        for ref in data["refs"]:
            points_refs.append(self.unsuperpoint(ref))
        outputs = {
            "points_tgt": points_tgt,
            "points_refs": points_refs,
        }
        return outputs

class UnsuperLoss():

    def __init__(self):
        pass

    def __call__(self, data):
        APrel = data["A"]["Prel"]
        AP = data["A"]["P"]
        AS = data["A"]["S"]
        AF = data["A"]["F"]
        BPrel = data["B"]["Prel"]
        BP = data["B"]["P"]
        BS = data["B"]["S"]
        BF = data["B"]["F"]
        homog = data["homography"]
        H, W = data["img"].shape[2:]

        #print("homog", homog[0], "AP", AP.shape)

        B, N = AS.shape

        T = torch.tensor([
            [1, 0, W/2.],
            [0, 1, H/2.],
            [0, 0, 1],
        ], dtype=torch.float).repeat(B,1,1).to(homog.device)
        
        # Points from branch A transformed by homography        
        APh = from_homog_coords(T @ homog @ torch.inverse(T) @ to_homog_coords(AP))

        # Matrix of distances between points
        D = (APh.permute(0,2,1).unsqueeze(2) - BP.permute(0,2,1).unsqueeze(1)).norm(dim=-1)

        # Create ids which maps the B points to its closest A point (A[ids] <-> B)
        Dmin, ids = torch.min(D, dim=1)

        # Create a mask for only the maped ids that are closer than a threshold.
        mask = Dmin.le(4)

        #print(mask[0].sum(), mask[1].sum(), mask[2].sum(), mask[3].sum())

        d = Dmin[mask]
        dmean = d.mean()

        # Distances between corresponding points should be small
        l_position = d

        mask_ = mask.view(-1)
        ids_ = ids.view(-1)
        AS_ = AS.view(-1)[ids_][mask_] # see debugger_point
        BS_ = BS.view(-1)[mask_]

        # Scores of corresonding points should be similar to each other
        l_score_sim = (AS_ - BS_) ** 2

        # Increase score if they are near, supress score if they are far
        S_ = (AS_ + BS_) / 2
        l_score_usp = S_ * (d - dmean)

        # Descriptor
        C = D.le(8)
        lam_d = 100
        mp = 1
        mn = 0.2
        af = AF.permute(0,2,1).unsqueeze(2)
        bf = BF.permute(0,2,1).unsqueeze(1)
        dot = (af * bf).sum(dim=-1) # [B,N,N]
        pos = torch.clamp(mp - dot, min=0)
        neg = torch.clamp(dot - mn, min=0)
        l_desc = (lam_d * C * pos + (~C) * neg)

        # Decorrelation
        l_decorr_a = decorrelate(AF)
        l_decorr_b = decorrelate(BF)
        l_decorr = 0.5 * (l_decorr_a + l_decorr_b)

        # Uniform distribution of relative positions
        l_uni_ax = uniform_distribution_loss(APrel[:,0,:])
        l_uni_ay = uniform_distribution_loss(APrel[:,1,:])
        l_uni_bx = uniform_distribution_loss(BPrel[:,0,:])
        l_uni_by = uniform_distribution_loss(BPrel[:,1,:])

        # Loss terms
        loss_position  = 1.0   * l_position.mean()
        loss_score_sim = 10.0  * l_score_sim.mean()
        loss_score_usp = 1.0   * l_score_usp.mean()
        loss_desc      = 1.0   * l_desc.mean()
        loss_decorr    = 10.0   * l_decorr.mean()
        loss_uni_xy    = 1000.0 * (l_uni_ax.mean() + l_uni_ay.mean() + l_uni_bx.mean() + l_uni_by.mean())

        #loss = 1 * loss_usp + 0.001 * loss_desc + 0.03 * loss_decorr + 100 * loss_uni_xy
        loss = loss_position + loss_score_sim + loss_score_usp + loss_desc + loss_decorr + loss_uni_xy

        print(loss_position.item(), loss_score_sim.item(), loss_score_usp.item(), loss_desc.item(), loss_decorr.item(), loss_uni_xy.item())

        return loss, { "ids": ids, "mask": mask, "APh": APh }

def main():
    dataset = HomoAdapDataset("/home/ai/Code/Data/coco/unlabeled2017/")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False,
        num_workers=1, pin_memory=True)

    loss_fn = UnsuperLoss()

    model = SiameseUnsuperPoint()
    model.train()

    optimizer = torch.optim.Adam(model.parameters())

    for inputs in loader:
        print(list(inputs.keys()))

        with torch.enable_grad():
            loss, data = utils.forward_pass(model, loss_fn, inputs)

        print(loss.item())

        brute_force_match(data["A"]["F"], data["B"]["F"])

        utils.backward_pass(optimizer, loss)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    main()