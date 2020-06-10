
import torch
import data
import utils
import viz
import numpy as np
import cv2
from networks.deepconsensus import FundamentalConsensus, FundamentalConsensusLoss
from matplotlib import pyplot as plt

class DebuggerFcons():

    def __init__(self, args):
        super().__init__()

        self.DEVICE = args.device

        self.loader = data.get_coco_batch_loader_split(args)["test"]

        self.model = FundamentalConsensus().to(self.DEVICE)

        self.loss_fn = FundamentalConsensusLoss()

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

        b = 0

        while True:

            print(f"b={b}")

            prel1 = utils.torch_to_numpy(data["A"]["Prel"][b].transpose(0,1))
            prel2 = utils.torch_to_numpy(data["B"]["Prel"][b].transpose(0,1))

            prelflat = np.concatenate((prel1.flatten(), prel2.flatten()))
            #prelflat = prel1[:,0]

            img = viz.tensor2img(data["img"][b])
            warp = viz.tensor2img(data["warp"][b])

            AF = data["A"]["F"]
            BF = data["B"]["F"]
            B = BF.shape[0]
            des1 = utils.torch_to_numpy(AF[b].transpose(0,1))
            des2 = utils.torch_to_numpy(BF[b].transpose(0,1))

            s1 = utils.torch_to_numpy(data["A"]["S"][b])
            s2 = utils.torch_to_numpy(data["B"]["S"][b])

            p1 = utils.torch_to_numpy(data["A"]["P"][b].transpose(0,1))
            p2 = utils.torch_to_numpy(data["B"]["P"][b].transpose(0,1))

            img_matches = []

            if True: # match descriptors using pytorch
                x = utils.torch_to_numpy(data["x"][b].transpose(0,1))
                w = utils.torch_to_numpy(data["w"][b])
                ap, bp = x[:,:2], x[:,2:]
                print(w.shape)
                inliers = (w > 0.99)
                img_matches.append(viz.draw_matches(img, warp, ap, bp, inliers))
            if True: # cv2 match descriptor
                bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(des1, des2)
                #matches = sorted(matches, key = lambda x: -x.distance)
                print(des1.shape, len(matches))
                #matches = matches[:20]
                kp1 = [cv2.KeyPoint(xy[0], xy[1], 2) for xy in p1]
                kp2 = [cv2.KeyPoint(xy[0], xy[1], 2) for xy in p2]
                img_matches.append(cv2.drawMatches(img, kp1, warp, kp2, matches, flags=2, outImg=None))
            
            img_matches = np.concatenate([img for img in img_matches])

            plt.ion()
            fig = plt.figure(1)
            #fig.show()
            plt.clf()

            while True:
                key = cv2.waitKey(10)
                if key == 27: # esc
                    print("exit")
                    exit()
                elif key == 119: # w
                    b = min(b+1, B-1)
                    break
                elif key == 115: # s
                    b = max(b-1, 0)
                    break
                elif key == 32: # space
                    print("next")
                    return
                
                #cv2.imshow("img", img)
                #cv2.imshow("warp", warp)
                cv2.imshow("matches", img_matches)

                #plt.hist(prelflat, 200, (0.,1.), color=(0,0,1))
                plt.hist(inliers, 10, (0.,1.), color=(0,0,1))
                fig.canvas.flush_events()
                