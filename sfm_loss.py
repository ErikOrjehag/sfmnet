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

        # Downscale t????
        K_scaled = torch.cat((K[:,:2]/downscale, K[:,2:]/downscale), dim=1)

        warps = []
        diffs = []

        for i, ref in enumerate(refs_scaled):
            pose = poses[:,i]

            ref_warped, valid_mask = inverse_warp(ref, depth, pose, K_scaled)[:2]
            
            diff = ((tgt_scaled - ref_warped) * valid_mask.unsqueeze(1).float()).abs()
            
            # Explainability
            exp = explain[:,i].unsqueeze(1)
            diff *= exp
            
            loss += diff.mean()
            warps.append(ref_warped)
            diffs.append(diff)

        #print(len(warps))
        #print(warps[0].shape)
        warps = torch.stack(warps, dim=1)
        #print(warps.shape)
        #exit()
        diffs = torch.stack(diffs, dim=1)


        return loss, warps, diffs

    total_loss = 0.0
    all_warps = []
    all_diffs = []

    assert len(depths) == len(explains)
    for depth, explain in zip(depths, explains):
        loss, warps, diffs = one_scale_loss(depth, explain)
        total_loss += loss
        all_warps.append(warps)
        all_diffs.append(diffs)

    return total_loss, all_warps, all_diffs

def smooth_loss(depths):
    def gradient(depth):
        D_dy = depth[:,:,1:] - depth[:,:,:-1]
        D_dx = depth[:,:,:,1:] - depth[:,:,:,:-1]
        return D_dx, D_dy
    loss = 0.0
    weight = 1.0
    for depth in depths:
        dx, dy = gradient(depth)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += weight * (
            dx2.abs().mean() + 
            dxdy.abs().mean() + 
            dydx.abs().mean() + 
            dy2.abs().mean())
        weight /= 2.3
    return loss

def explainability_loss(masks):
    loss = 0
    for i, mask in enumerate(masks):
        ones = torch.ones_like(mask)
        loss += F.binary_cross_entropy(mask, ones)
    return loss

class SfmLoss():
    def __init__(self):
        pass

    def __call__(self, inputs, outputs):
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
        
        lr, warps, diffs = photometric_reconstruction_loss(tgt, refs, K, depths, explains, poses)
        ls = smooth_loss(depths)
        le = explainability_loss(explains)

        loss = lr + 0.5 * ls + 0.2 * le

        return loss, warps, diffs