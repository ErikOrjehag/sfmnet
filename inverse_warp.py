import torch
import torch.nn.functional as F
from geometry import euler2mat

_pixel_coord_cache = None

def pixel2cam(depth, Kinv):
    global _pixel_coord_cache
    B, H, W = depth.shape

    if (_pixel_coord_cache is None) \
        or _pixel_coord_cache.shape[2] < H \
        or _pixel_coord_cache.shape[3] < W:
        x = torch.arange(0, H).view(1, H, 1).expand(1, H, W).type_as(depth)
        y = torch.arange(0, W).view(1, 1, W).expand(1, H, W).type_as(depth)
        ones = torch.ones(1, H, W).type_as(depth)
        _pixel_coord_cache = torch.stack((x, y, ones), dim=1)

    # [B, 3, H*W]
    pixel_coords = _pixel_coord_cache[...,:H,:W].expand(B, 3, H, W).reshape(B, 3, -1)
    cam_coords = (Kinv @ pixel_coords).reshape(B, 3, H, W)
    return cam_coords * depth.unsqueeze(1)



def pose_vec2mat(vec):
    t = vec[:,:3].unsqueeze(-1) # [B, 3, 1]
    r = vec[:,3:]
    R = euler2mat(r) # [B, 3, 3]
    transform = torch.cat([R, t], dim=2) # [B, 3, 4]
    return transform

def inverse_warp(img, depth, pose, K):

    B, _, H, W = img.shape

    cam_coords = pixel2cam(depth, K.inverse())

    T = pose_vec2mat(pose)

    C = K @ T # [B, 3, 4]

    pixel_coords = cam2pixel(cam_coords, C) # [B, H, W, 2]
    
    
    #projected = F.grid_sample(img, pixel_coords, padding_mode="zeros")

    #valid_pts = pixel_coords.abs().max(dim=-1)[0] <= 1

    print(T)

    exit()
