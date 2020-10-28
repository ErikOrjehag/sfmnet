
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
from networks.deepconsensus import HomographyConsensus, HomographyConsensusLoss
from point_tester import calc_metrics

class HomographyTester():

    def __init__(self, args):
        super().__init__()

        self.DEVICE = args.device

        self.loader = data.get_batch_loader_split(args)["test"]

        print("Test images: ", len(self.loader))

        self.model = HomographyConsensus().to(args.device)

        self.loss_fn = HomographyConsensusLoss(pred=True)

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
        Hpred = data["Hpred"]
        AP = data["A"]["Pmax"]
        BP = data["B"]["Pmax"]
        AF = data["A"]["Fmax"]
        BF = data["B"]["Fmax"]
        N = AP.shape[1]

        Aimg = viz.tensor2img(data["img"][0])
        Bimg = viz.tensor2img(data["warp"][0])

        unsup_RS, unsup_LE, unsup_MS, unsup_TP, unsup_N, unsup_Nm = calc_metrics(homog, AP, BP, AF, BF, W, H, 'NORM_L2')
        HA = calc_HA(homog, Hpred)

        print(f"{step}/{len(self.loader)-1} - \n" + 
            f"\tunsup_RS={unsup_RS:.2f}, unsup_LE={unsup_LE:.2f}, unsup_MS={unsup_MS:.2f}, unsup_TP={unsup_TP:.2f}, unsup_N={unsup_N:.2f}, unsup_Nm={unsup_Nm:.2f}")

        metrics = {
            "unsup_RS": unsup_RS,
            "unsup_LE": unsup_LE,
            "unsup_MS": unsup_MS,
            "unsup_TP": unsup_TP,
            "unsup_N": unsup_N,
            "unsup_Nm": unsup_Nm
        }

        self.metrics = utils.dict_append(self.metrics, metrics)

def calc_HA(Hgt, Hpred):
    pass
