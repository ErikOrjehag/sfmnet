
import torch
import data
import utils
import viz
import numpy as np
import cv2
from networks.unsuperpoint import SiameseUnsuperPoint, UnsuperLoss, brute_force_match, SequenceUnsuperPoint
from matplotlib import pyplot as plt
from networks.deepconsensus import HomographyConsensusSynthPoints, HomographyConsensusSynthPointsLoss, HomographyConsensusLoss

class DebuggerBase():
    def __init__(self, args):

        self.DEVICE = args.device

        self.loader = data.get_batch_loader_split(args)["test"]

        self.model, self.loss_fn = self._setup_model_and_loss()

        if args.load:
            checkpoint = torch.load(args.load, map_location=torch.device(args.device))
            self.model.load_state_dict(checkpoint["model"])
        else:
            if args.load_consensus:
                model_dict = self.model.state_dict()
                consensus_checkpoint = torch.load(args.load_consensus, map_location=torch.device(args.device))
                model_dict.update(consensus_checkpoint["model"])
                self.model.load_state_dict(model_dict)
            if args.load_point:
                point_checkpoint = torch.load(args.load_point, map_location=torch.device(args.device))
                self.model.siamese_unsuperpoint.load_state_dict(point_checkpoint["model"])


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

class DebuggerHomoSynthPoint(DebuggerBase):

    def __init__(self, args):
        super().__init__(args)

        self.b = 0 # current batch for rendering

    def _setup_model_and_loss(self):
        return (
            HomographyConsensusSynthPoints().to(self.DEVICE), 
            HomographyConsensusLoss(pred=True) 
        )

    def _debug_step(self, loss, data):

        B = data["p"].shape[0]
        while True:
            print(f"b = {self.b}")
            #p = utils.torch_to_numpy(data["p"])[self.b].transpose(0,1)
            #ph = utils.torch_to_numpy(data["ph"])[self.b].transpose(0,1)

            x = utils.torch_to_numpy(data["x"][self.b].transpose(0,1))
            ap, bp = x[:,:2], x[:,2:]
            H = 128
            W = 416
            ap[:,0] += W/2
            ap[:,1] += H/2
            bp[:,0] += W/2
            bp[:,1] += H/2
            
            w_gt = utils.torch_to_numpy(data["w_gt"])[self.b]
            inliers_gt = w_gt > 0.5

            w = utils.torch_to_numpy(data["w"])[self.b]
            #inlier_sig = utils.torch_to_numpy(data["inlier_sig"])[self.b]
            inliers = w > 0.5

            N = ap.shape[0]

            img_a = np.uint8(np.zeros((H, W, 3)))
            img_b = np.uint8(np.zeros((H, W, 3)))

            img_matches = []

            img_matches.append(viz.draw_text("dataset", viz.draw_matches(img_a, img_b, ap, bp, inliers_gt)))
            img_matches.append(viz.draw_text("pred", viz.draw_matches(img_a, img_b, ap, bp, inliers)))

            img_matches = np.concatenate([img for img in img_matches])

            plt.ion()
            fig = plt.figure(1)
            plt.clf()

            while True:
                key = cv2.waitKey(10)
                if key == 27: # esc
                    print("exit")
                    exit()
                elif key == 119: # w
                    self.b = min(self.b+1, B-1)
                    break
                elif key == 115: # s
                    self.b = max(self.b-1, 0)
                    break
                elif key == 32: # space
                    print("next")
                    return
                
                cv2.imshow("matches", img_matches)

                #plt.hist(self.prelflat, 200, (0.,1.), color=(0,0,1))
                plt.hist(w, 10, (0.,1.), color=(0,0,1))
                #plt.hist(inlier_sig, 10, (0.,1.), color=(0,0,1))
                
                fig.canvas.flush_events()




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

            self.AF = data["A"]["Fmax"]
            self.BF = data["B"]["Fmax"]
            self.B = self.BF.shape[0]
            self.des1 = utils.torch_to_numpy(self.AF[self.b].transpose(0,1))
            self.des2 = utils.torch_to_numpy(self.BF[self.b].transpose(0,1))

            self.s1 = utils.torch_to_numpy(data["A"]["S"][self.b])
            self.s2 = utils.torch_to_numpy(data["B"]["S"][self.b])

            self.p1 = utils.torch_to_numpy(data["A"]["Pmax"][self.b].transpose(0,1))
            self.p2 = utils.torch_to_numpy(data["B"]["Pmax"][self.b].transpose(0,1))

            self.p1_all = utils.torch_to_numpy(data["A"]["P"][self.b].transpose(0,1))
            self.p2_all = utils.torch_to_numpy(data["B"]["P"][self.b].transpose(0,1))

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

                plt.hist(self.prelflat, 20, (0.,1.), color=(0,0,1))
                #plt.hist(self.inliers, 10, (0.,1.), color=(0,0,1))
                #plt.hist(self.s1, 100, (0.,1.), color=(0,0,1))
                
                fig.canvas.flush_events()



class PointDebugger(DebuggerPointBase):

    def __init__(self, args):
        super().__init__(args)

    def _setup_model_and_loss(self):
        return (
            SiameseUnsuperPoint().to(self.DEVICE), 
            #UnsuperLoss() 
            lambda x: (torch.rand(1), x),
        )

    def _compute_debug(self, loss, data):

        if True: # match descriptors using pytorch
            ids, mask = brute_force_match(self.AF, self.BF)
            ids = utils.torch_to_numpy(ids[self.b])
            mask = utils.torch_to_numpy(mask[self.b])
            print(self.p1[mask].shape)
            self.img_matches.append(viz.draw_text("PyTorch Matcher", viz.draw_matches(self.img, self.warp, self.p1[mask], self.p2[ids][mask])))
            print(ids.shape, mask.sum())

        if False: # debug match using ids
            ids = utils.torch_to_numpy(data["ids"][self.b])
            mask = utils.torch_to_numpy(data["mask"][self.b])
            APh = utils.torch_to_numpy(data["APh"][self.b])
            p1h = utils.torch_to_numpy(data["APh"][self.b].transpose(0,1))
            self.img_matches.append(viz.draw_text("Matched ids", viz.draw_matches(self.img, self.warp, self.p1_all[ids][mask], self.p2_all[mask])))
            #self.img_matches.append(viz.draw_matches(img, warp, p1, p1h))

class SynthHomoPointDebugger(DebuggerBase):


    def __init__(self, args):
        super().__init__(args)

        self.b = 0 # current batch for rendering

    def _debug_step(self, loss, data):

        while True:
            print(f"b = {self.b}")

            

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

                #plt.hist(self.prelflat, 200, (0.,1.), color=(0,0,1))
                plt.hist(self.inliers, 10, (0.,1.), color=(0,0,1))
                
                fig.canvas.flush_events()