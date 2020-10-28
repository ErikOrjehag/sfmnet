
import torch
import data
import utils
import viz
import numpy as np
import cv2
from networks.deepconsensus import HomographyConsensus, HomographyConsensusLoss
from matplotlib import pyplot as plt
from point_debugger import DebuggerPointBase

class HomographyDebugger(DebuggerPointBase):

    def __init__(self, args):
        super().__init__(args)

    def _setup_model_and_loss(self):
        return (
            HomographyConsensus().to(self.DEVICE), 
            HomographyConsensusLoss(pred=True)
        )

    def _compute_debug(self, loss, data):

        if True: # match descriptors using pytorch
            x = utils.torch_to_numpy(data["x"][self.b].transpose(0,1))
            ap, bp = x[:,:2], x[:,2:]
            H = 128
            W = 416
            ap *= W
            bp *= W
            ap[:,0] += W/2
            ap[:,1] += H/2
            bp[:,0] += W/2
            bp[:,1] += H/2
            
            w = utils.torch_to_numpy(data["w"][self.b])
            self.inliers = w
            
            print(w.shape)
            inliers = (w > 0.5)
            self.img_matches.append(viz.draw_text("inliers fcons", viz.draw_matches(self.img, self.warp, ap, bp, inliers)))
            #self.img_matches.append(np.concatenate((self.img, self.warp), axis=1))

            src_pts = np.float32(ap).reshape(-1,1,2)
            dst_pts = np.float32(bp).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            mask = mask.flatten().astype(np.bool)
            n_cv2_inliers = mask.sum()
            n_cv2_total = mask.shape[0]
            print(f"cv2 inliers/outliers/total: ${n_cv2_inliers}/${n_cv2_total-n_cv2_inliers}/${n_cv2_total} -> ${100.0*(n_cv2_inliers/n_cv2_total)}%")
            self.img_matches.append(viz.draw_text("CV2 find homo", viz.draw_matches(self.img, self.warp, ap, bp, mask)))

            img_warped = cv2.warpPerspective(self.img, M, (self.img.shape[1], self.img.shape[0]))
            ap_warped = cv2.perspectiveTransform(src_pts, M).squeeze(1)
            self.img_matches.append(viz.draw_text("CV2 warp", viz.draw_matches(img_warped, self.warp, ap_warped, bp, mask, draw_outliers=False)))

            T = np.array([
                [W, 0, W/2.0],
                [0, W, H/2.0],
                [0, 0, 1.0],
            ])

            H_pred = utils.torch_to_numpy(data["H_pred"][self.b].inverse())
            H_pred = T @ H_pred @ np.linalg.inv(T)
            H = H_pred / H_pred[2,2]

            img_H_warped = cv2.warpPerspective(self.img, H, (self.img.shape[1], self.img.shape[0]))
            ap_H_warped = cv2.perspectiveTransform(src_pts, H).squeeze(1)
            self.img_matches.append(viz.draw_text("H warp", viz.draw_matches(img_H_warped, self.warp, ap_H_warped, bp, mask, draw_outliers=False)))

