
import torch
import data
import utils
import viz
import numpy as np
import cv2
from networks.unsuperpoint import SiameseUnsuperPoint, UnsuperLoss


class DebuggerPoint():

    def __init__(self, args):
        super().__init__()

        self.DEVICE = args.device

        self.loader = data.get_coco_loader(args)

        self.model = SiameseUnsuperPoint().to(self.DEVICE)

        self.loss_fn = UnsuperLoss()

        checkpoint = torch.load(args.load, map_location=torch.device(args.device))
        self.model.load_state_dict(checkpoint["model"])

    def run(self):

        self.model.eval()

        utils.iterate_loader(self.DEVICE, self.loader, self._step_fn)

    def _step_fn(self, step, inputs):

        # Forward pass and loss
        with torch.no_grad():
            loss, data = utils.forward_pass(self.model, self.loss_fn, inputs)

        print(f"loss {loss.item():.3f}")

        img = viz.tensor2img(data["img"][0])
        warp = viz.tensor2img(data["warp"][0])

        des1 = data["A"]["F"][0].transpose(0,1).detach().cpu().numpy()
        des2 = data["B"]["F"][0].transpose(0,1).detach().cpu().numpy()

        bf = cv2.BFMatcher(cv2.DIST_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)

        p1 = data["A"]["P"][0].transpose(0,1).detach().cpu().numpy()
        p2 = data["B"]["P"][0].transpose(0,1).detach().cpu().numpy()

        kp1 = [cv2.KeyPoint(xy[0], xy[1], 2) for xy in p1]
        kp2 = [cv2.KeyPoint(xy[0], xy[1], 2) for xy in p2]

        img_matches = cv2.drawMatches(img, kp1, warp, kp2, matches, flags=2, outImg=None)

        loop = True
        while loop:
            key = cv2.waitKey(10)
            if key == 27:
                exit()
            elif key != -1:
                loop = False
            
            #cv2.imshow("img", img)
            #cv2.imshow("warp", warp)
            cv2.imshow("matches", img_matches)
            