import torch
import torch.nn.functional as F
from inverse_warp import inverse_warp

def photometric_reconstruction_loss(tgt, refs, K, depths, explains, poses):
    
    def one_scale_loss(depth, explain):
        loss = 0
        B, _, H, W = depth.shape
        downscale = tgt.shape[2] / H

        tgt_scaled = F.interpolate(tgt, (H, W), mode="area")
        refs_scaled = [F.interpolate(ref.squeeze(1), (H, W), mode="area") for ref in torch.split(refs, 1, dim=1)]

        K_scaled = torch.cat((K[:,:2]/downscale, K[:,2:]), dim=1)

        for i, ref in enumerate(refs_scaled):
            pose = poses[:,i]
            ref_warped, valid_mask = inverse_warp(ref, depth.squeeze(1), pose, K_scaled)
            diff = (tgt_scaled - ref_warped) # * valid_mask.unsqueeze(1).float()
            loss += diff.abs().mean()
        return loss

    total_loss = sum([one_scale_loss(depth, explain) 
        for depth, explain in zip(depths, explains)])

    return total_loss

class SfmLoss():
    def __init__(self):
        pass

    def __call__(self, outputs, inputs):
        tgt, refs, K, Kinv = inputs
        (depths), (poses, explains) = outputs

        #print(H, W)
        #print(len(depths))
        #print(depths[0].shape)
        #print(poses.shape)
        #print(explains[0].shape)
        #print(tgt.shape)
        #print(refs.shape)
        #print(K.shape)
        #print(K.inverse())
        #print(Kinv)
        
        loss = photometric_reconstruction_loss(tgt, refs, K, depths, explains, poses)

        return loss