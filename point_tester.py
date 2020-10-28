
import torch
import sfm_loss
import data
import networks.architectures
import utils
import numpy as np
import reconstruction
import geometry
import evaluation
from networks.unsuperpoint import SiameseUnsuperPoint
from geometry import from_homog_coords, to_homog_coords
import cv2
import viz
from networks.unsuperpoint import brute_force_match

def no_loss(data):
    return torch.tensor(0.0), {}

# RS, LE, MS

class PointTester():

    def __init__(self, args):
        super().__init__()

        self.DEVICE = args.device

        self.loader = data.get_batch_loader_split(args)["test"]

        print("Test images: ", len(self.loader))

        self.model = SiameseUnsuperPoint().to(args.device)

        self.loss_fn = no_loss

        checkpoint = torch.load(args.load, map_location=torch.device(args.device))
        self.model.load_state_dict(checkpoint["model"])

        self.metrics = {}

    def run(self):

        self.model.eval()

        utils.iterate_loader(self.DEVICE, self.loader, self._step_fn, end=len(self.loader))

        mean_metrics = utils.dict_mean(self.metrics)
        std_metrics = utils.dict_std(self.metrics)

        print("\n"+"="*20)
        print("Metrics:")
        for key in self.metrics:
            print(f"{key} -> mean: {mean_metrics[key]:0.3f}, std: {std_metrics[key]:0.3f}")
        print("="*20)

    def _step_fn(self, step, inputs):

        # Forward pass and loss
        with torch.no_grad():
            loss, data = utils.forward_pass(self.model, self.loss_fn, inputs)

        H, W = data["img"].shape[2:]
        homog = data["homography"]
        AP = data["A"]["Pmax"]
        BP = data["B"]["Pmax"]
        AF = data["A"]["Fmax"]
        BF = data["B"]["Fmax"]
        N = AP.shape[1]

        Aimg = viz.tensor2img(data["img"][0])#cv2.cvtColor(viz.tensor2img(data["img"][0]), cv2.COLOR_BGR2GRAY)
        Bimg = viz.tensor2img(data["warp"][0])#cv2.cvtColor(viz.tensor2img(data["warp"][0]), cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        orb_AP, orb_AF = orb_detect(orb, Aimg, device=AP.device)
        orb_BP, orb_BF = orb_detect(orb, Bimg, device=AP.device)

        unsup_RS, unsup_LE, unsup_MS, unsup_CMR, unsup_N, unsup_Nm = calc_metrics(homog, AP, BP, AF, BF, W, H, 'NORM_L2')
        orb_RS, orb_LE, orb_MS, orb_CMR, orb_N, orb_Nm = calc_metrics(homog, orb_AP, orb_BP, orb_AF, orb_BF, W, H, 'NORM_HAMMING')

        print(f"{step}/{len(self.loader)-1} - \n" + 
            f"\tunsup_RS={unsup_RS:.2f}, unsup_LE={unsup_LE:.2f}, unsup_MS={unsup_MS:.2f}, unsup_CMR={unsup_CMR:.2f}, unsup_N={unsup_N:.2f}, unsup_Nm={unsup_Nm:.2f}\n" +
            f"\torb_RS={orb_RS:.2f}, orb_LE={orb_LE:.2f}, orb_MS={orb_MS:.2f}, orb_CMR={orb_CMR:.2f}, orb_N={orb_N:.2f}, orb_Nm={orb_Nm:.2f}")

        metrics = {
            "unsup_RS": unsup_RS,
            "unsup_LE": unsup_LE,
            "unsup_MS": unsup_MS,
            "unsup_CMR": unsup_CMR,
            "unsup_N": unsup_N,
            "unsup_Nm": unsup_Nm,
            "orb_RS": orb_RS,
            "orb_LE": orb_LE,
            "orb_MS": orb_MS,
            "orb_CMR": orb_CMR,
            "orb_N": orb_N,
            "orb_Nm": orb_Nm,
        }

        self.metrics = utils.dict_append(self.metrics, metrics)

def calc_metrics(homog, AP, BP, AF, BF, W, H, normType):

    phi = 3
    pad = 10

    T = torch.tensor([
        [1, 0, W/2.],
        [0, 1, H/2.],
        [0, 0, 1],
    ], dtype=torch.float).repeat(1, 1, 1).to(AP.device)

    # Points from branch A transformed by homography        
    APh = from_homog_coords(T @ homog @ torch.inverse(T) @ to_homog_coords(AP))
    BPh = from_homog_coords(T @ torch.inverse(homog) @ torch.inverse(T) @ to_homog_coords(BP))

    Amask = torch.logical_and(
        torch.logical_and(
            torch.logical_and(APh[0,0] > pad, APh[0,1] > pad), 
            torch.logical_and(APh[0,0] < W-pad, APh[0,1] < H-pad),
        ),
        torch.logical_and(
            torch.logical_and(AP[0,0] > pad, AP[0,1] > pad), 
            torch.logical_and(AP[0,0] < W-pad, AP[0,1] < H-pad),
        ),
    )

    Bmask = torch.logical_and(
        torch.logical_and(
            torch.logical_and(BPh[0,0] > pad, BPh[0,1] > pad), 
            torch.logical_and(BPh[0,0] < W-pad, BPh[0,1] < H-pad),
        ),
        torch.logical_and(
            torch.logical_and(BP[0,0] > pad, BP[0,1] > pad), 
            torch.logical_and(BP[0,0] < W-pad, BP[0,1] < H-pad),
        ),
    )

    APhm = APh[...,Amask]

    BPm = BP[...,Bmask]

    # Matrix of distances between points
    D = (APhm.permute(0,2,1).unsqueeze(2) - BPm.permute(0,2,1).unsqueeze(1)).norm(dim=-1)

    maxi = np.argmax(D.shape[1:])

    Dmin, Dids = torch.min(D, dim=maxi+1)
    Dmask = Dmin.le(phi)
    Dmask_sum = Dmask.sum()
    Ntot = Dmask.shape[-1]
    RS = torch.true_divide(Dmask_sum, Ntot).item()

    LE = Dmin[Dmask].mean().item()

    AFm = AF[...,Amask]
    BFm = BF[...,Bmask]
    
    if maxi == 1:
        f1 = AFm
        f2 = BFm
    else:
        f1 = BFm
        f2 = AFm

    Fids, crossCheckMask = brute_force_match(f1, f2, normType)
    
    n_crossCheckMask = crossCheckMask.sum().item()

    mask_has_desc_match_and_are_close = torch.logical_and(Dmask, crossCheckMask)
    n_has_desc_match_and_are_close = mask_has_desc_match_and_are_close.sum()

    # Ratio of matched points that are correct.
    CMRn = torch.logical_and(Fids == Dids, mask_has_desc_match_and_are_close).sum()
    CMR = torch.true_divide(CMRn, n_has_desc_match_and_are_close).item()
    
    # Ratio between correctly matched points and all points in the image
    MS = torch.true_divide(CMRn, Ntot).item()

    return RS, LE, MS, CMR, float(Ntot), float(n_crossCheckMask)

def orb_detect(orb, img, device):
    orb_kp, orb_desc = orb.detectAndCompute(img, None)
    orb_P = torch.tensor([kp.pt for kp in orb_kp]).transpose(0,1).unsqueeze(0).to(device)
    orb_F = torch.tensor(np.unpackbits(orb_desc, axis=1), dtype=torch.bool).transpose(0,1).unsqueeze(0).to(device)
    return orb_P, orb_F