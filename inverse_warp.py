import torch
import torch.nn.functional as F
from geometry import euler2mat

_homog_pixel_coord_cache = None

def create_homog_pixel_coords(B, H, W, typ):
    global _homog_pixel_coord_cache
    if (_homog_pixel_coord_cache is None) \
        or _homog_pixel_coord_cache.shape[2] < H \
        or _homog_pixel_coord_cache.shape[3] < W:
        y = torch.arange(0, H).view(1, H, 1).expand(1, H, W).type(typ)
        x = torch.arange(0, W).view(1, 1, W).expand(1, H, W).type(typ)
        ones = torch.ones(1, H, W).type(typ)
        _homog_pixel_coord_cache = torch.stack((x, y, ones), dim=1)
    return _homog_pixel_coord_cache[...,:H,:W].expand(B, 3, H, W)
    
def multiply_coords(matrix, coords):
    B, C, H, W = coords.shape
    flat = coords.reshape(B, C, -1) # [B,C,H*W]
    res = (matrix @ flat)
    res = res.reshape(B, -1, H, W)
    return res
 
def pose_vec2mat(vec):
    t = vec[:,:3].unsqueeze(-1) # [B, 3, 1]
    r = vec[:,3:]
    R = euler2mat(r) # [B, 3, 3]
    transform = torch.cat([R, t], dim=2) # [B, 3, 4]
    return transform

def to_homog_coords(coords):
    return F.pad(input=coords, pad=(0, 0, 0, 0, 0, 1), mode="constant", value=1)

def to_homog_matrix(matrix):
    matrix = F.pad(input=matrix, pad=(0, 0, 0, 1), mode="constant", value=0)
    matrix[...,-1,-1] = 1.0
    return matrix

def from_homog_coords(coords):
    X = coords[:,0]
    Y = coords[:, 1]
    Z = coords[:, 2].clamp(min=1e-3)
    return torch.stack((X/Z, Y/Z), dim=1)
    
def to_sampling_grid(coords):
    B, C, H, W = coords.shape
    # -1 extreme left, +1 extreme right
    print("C", C)
    flat = coords.reshape(B, C, -1) # [B,2,H*W]
    print("flat", flat.shape)
    X = 2*flat[:, 0] / (W-1) - 1
    Y = 2*flat[:, 1] / (H-1) - 1
    print("x shape", X.shape)
    print("y shape", Y.shape)
    p = torch.stack((X, Y), dim=2)
    print("p", p.shape)
    return p.reshape(B, H, W, 2)

def pad_zero_column_right(K):
    return F.pad(input=K, pad=(0, 1), mode="constant", value=0)

def inverse_warp(img, depth, pose, K):

    B, _, H, W = img.shape
    Kinv = K.inverse()

    T = pose_vec2mat(pose)
    D = depth.unsqueeze(1)

    # Homogeneous pixel coordinates
    homog_pixel_coords = create_homog_pixel_coords(B, H, W, img.type())
    
    # Rays shooting out from target frame
    rays = multiply_coords(Kinv, homog_pixel_coords)
    
    # Points hitting objects out in the world
    homog_world_points = to_homog_coords(D * rays)

    # The world points projected back into the reference view
    KT = pad_zero_column_right(K) @ to_homog_matrix(T)
    #print(KT)
    homog_projected_pixel_coords = multiply_coords(KT, homog_world_points)

    # Normalize homogeneous pixel coordinates
    projected_pixel_coords = from_homog_coords(homog_projected_pixel_coords)

    # Sample the source image in the projected pixel coordinates
    sampling_grid = to_sampling_grid(projected_pixel_coords)

    reconstruction = F.grid_sample(img, sampling_grid, padding_mode="zeros", align_corners=False)

    # Sampling points with abs value smaller than 1 are inside the frame
    valid_mask = sampling_grid.abs().max(dim=-1)[0] <=1

    return reconstruction , valid_mask, homog_pixel_coords, rays, homog_world_points, projected_pixel_coords, sampling_grid
