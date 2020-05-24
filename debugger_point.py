
import torch
import data
import utils
import viz
import numpy as np
import cv2
from networks.unsuperpoint import SiameseUnsuperPoint, UnsuperLoss, brute_force_match
from matplotlib import pyplot as plt

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
        print(list(data.keys()))

        print(f"loss {loss.item():.3f}")

        prel1 = utils.torch_to_numpy(data["A"]["Prel"][0].transpose(0,1))
        prel2 = utils.torch_to_numpy(data["B"]["Prel"][0].transpose(0,1))

        prelflat = np.concatenate((prel1.flatten(), prel2.flatten()))
        #prelflat = prel1[:,0]

        img = viz.tensor2img(data["img"][0])
        warp = viz.tensor2img(data["warp"][0])

        AF = data["A"]["F"]
        BF = data["B"]["F"]
        des1 = utils.torch_to_numpy(AF[0].transpose(0,1))
        des2 = utils.torch_to_numpy(BF[0].transpose(0,1))

        s1 = utils.torch_to_numpy(data["A"]["S"][0])
        s2 = utils.torch_to_numpy(data["B"]["S"][0])

        p1 = utils.torch_to_numpy(data["A"]["P"][0].transpose(0,1))
        p2 = utils.torch_to_numpy(data["B"]["P"][0].transpose(0,1))

        img_matches = []

        if True: # match descriptors using pytorch
            ids, mask = brute_force_match(AF, BF)
            ids = utils.torch_to_numpy(ids[0])
            mask = utils.torch_to_numpy(mask[0])
            img_matches.append(viz.draw_matches(img, warp, p1[mask], p2[ids][mask]))
            print(ids.shape, mask.sum())
        if True: # cv2 match descriptor
            bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des1, des2)
            #matches = sorted(matches, key = lambda x: -x.distance)
            print(des1.shape, len(matches))
            #matches = matches[:20]
            kp1 = [cv2.KeyPoint(xy[0], xy[1], 2) for xy in p1]
            kp2 = [cv2.KeyPoint(xy[0], xy[1], 2) for xy in p2]
            img_matches.append(cv2.drawMatches(img, kp1, warp, kp2, matches, flags=2, outImg=None))
        if False: # debug match using ids
            ids = utils.torch_to_numpy(data["ids"][0])
            mask = utils.torch_to_numpy(data["mask"][0])
            APh = utils.torch_to_numpy(data["APh"][0])
            p1h = utils.torch_to_numpy(data["APh"][0].transpose(0,1))
            img_matches.append(viz.draw_matches(img, warp, p1[ids][mask], p2[mask]))
            #img_matches.append(viz.draw_matches(img, warp, p1, p1h))
        
        img_matches = np.concatenate([img for img in img_matches])

        plt.ion()
        fig = plt.figure(1)
        #fig.show()
        plt.clf()

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

            plt.hist(prelflat, 200, (0.,1.), color=(0,0,1))
            fig.canvas.flush_events()
            