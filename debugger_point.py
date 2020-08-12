
import torch
import data
import utils
import viz
import numpy as np
import cv2
from networks.unsuperpoint import SiameseUnsuperPoint, UnsuperLoss, brute_force_match
from matplotlib import pyplot as plt

class DebuggerBase():
    def __init__(self, args):

        self.DEVICE = args.device

        self.loader = data.get_coco_batch_loader_split(args)["test"]

        self.model, self.loss_fn = self._setup_model_and_loss()

        checkpoint = torch.load(args.load, map_location=torch.device(args.device))
        self.model.load_state_dict(checkpoint["model"])

    def _setup_model_and_loss(self):
        pass

    def _debug_step(self, loss, data):
        pass

    def run(self):
        self.model.eval()
        utils.iterate_loader(self.DEVICE, self.loader, self._step_fn)

    def _step_fn(self, step, inputs):

        # Forward pass and loss
        with torch.no_grad():
            loss, data = utils.forward_pass(self.model, self.loss_fn, inputs)
        print(list(data.keys()))

        print(f"loss {loss.item():.3f}")

        self._debug_step(loss, data)

class DebuggerPointBase(DebuggerBase):

    def __init__(self, args):
        super().__init__(args)

        self.b = 0 # current batch for rendering

    def _compute_debug(self):
        pass

    def _debug_step(self, loss, data):

        while True:
            print(f"b = {self.b}")

            self.prel1 = utils.torch_to_numpy(data["A"]["Prel"][self.b].transpose(0,1))
            self.prel2 = utils.torch_to_numpy(data["B"]["Prel"][self.b].transpose(0,1))

            self.prelflat = np.concatenate((self.prel1.flatten(), self.prel2.flatten()))
            
            self.img = viz.tensor2img(data["img"][self.b])
            self.warp = viz.tensor2img(data["warp"][self.b])

            self.AF = data["A"]["F"]
            self.BF = data["B"]["F"]
            self.B = self.BF.shape[0]
            self.des1 = utils.torch_to_numpy(self.AF[self.b].transpose(0,1))
            self.des2 = utils.torch_to_numpy(self.BF[self.b].transpose(0,1))

            self.s1 = utils.torch_to_numpy(data["A"]["S"][self.b])
            self.s2 = utils.torch_to_numpy(data["B"]["S"][self.b])

            self.p1 = utils.torch_to_numpy(data["A"]["P"][self.b].transpose(0,1))
            self.p2 = utils.torch_to_numpy(data["B"]["P"][self.b].transpose(0,1))

            self.img_matches = []
            self.inliers = []

            if True: # cv2 match descriptor
                bf = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(self.des1, self.des2)
                #matches = sorted(matches, key = lambda x: -x.distance)
                print(self.des1.shape, len(matches))
                #matches = matches[:20]
                kp1 = [cv2.KeyPoint(xy[0], xy[1], 2) for xy in self.p1]
                kp2 = [cv2.KeyPoint(xy[0], xy[1], 2) for xy in self.p2]
                self.img_matches.append(viz.draw_text("CV2 BFMatcher", cv2.drawMatches(self.img, kp1, self.warp, kp2, matches, flags=2, outImg=None)))

            self._compute_debug(loss, data)

            self.img_matches = np.concatenate([img for img in self.img_matches])

            plt.ion()
            fig = plt.figure(1)
            plt.clf()

            while True:
                key = cv2.waitKey(10)
                if key == 27: # esc
                    print("exit")
                    exit()
                elif key == 119: # w
                    self.b = min(self.b+1, self.B-1)
                    break
                elif key == 115: # s
                    self.b = max(self.b-1, 0)
                    break
                elif key == 32: # space
                    print("next")
                    return
                
                cv2.imshow("matches", self.img_matches)

                plt.hist(self.prelflat, 200, (0.,1.), color=(0,0,1))
                #plt.hist(inliers, 10, (0.,1.), color=(0,0,1))
                
                fig.canvas.flush_events()



class DebuggerPoint(DebuggerPointBase):

    def __init__(self, args):
        super().__init__(args)

    def _setup_model_and_loss(self):
        return (
            SiameseUnsuperPoint().to(self.DEVICE), 
            UnsuperLoss() 
        )

    def _compute_debug(self, loss, data):

        if True: # match descriptors using pytorch
            ids, mask = brute_force_match(self.AF, self.BF)
            ids = utils.torch_to_numpy(ids[self.b])
            mask = utils.torch_to_numpy(mask[self.b])
            print(self.p1[mask].shape)
            self.img_matches.append(viz.draw_text("PyTorch Matcher", viz.draw_matches(self.img, self.warp, self.p1[mask], self.p2[ids][mask])))
            print(ids.shape, mask.sum())

        if True: # debug match using ids
            ids = utils.torch_to_numpy(data["ids"][self.b])
            mask = utils.torch_to_numpy(data["mask"][self.b])
            APh = utils.torch_to_numpy(data["APh"][self.b])
            p1h = utils.torch_to_numpy(data["APh"][self.b].transpose(0,1))
            self.img_matches.append(viz.draw_text("Matched ids", viz.draw_matches(self.img, self.warp, self.p1[ids][mask], self.p2[mask])))
            #self.img_matches.append(viz.draw_matches(img, warp, p1, p1h))