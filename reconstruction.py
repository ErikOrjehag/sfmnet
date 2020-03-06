import torch
import torch.nn.functional as F
from geometry import euler2mat, pose_vec2mat, to_homog_coords, to_homog_matrix, from_homog_coords

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
     
def to_sampling_grid(coords):
    B, C, H, W = coords.shape
    # -1 extreme left, +1 extreme right
    flat = coords.reshape(B, C, -1) # [B,2,H*W]
    X = 2*flat[:, 0] / (W-1) - 1
    Y = 2*flat[:, 1] / (H-1) - 1
    p = torch.stack((X, Y), dim=2)
    return p.reshape(B, H, W, 2)

def pad_zero_column_right(K):
    return F.pad(input=K, pad=(0, 1), mode="constant", value=0)

def depth_to_3d_points(depth, K):
    B, _, H, W = depth.shape
    Kinv = K.inverse()

    # Homogeneous pixel coordinates
    homog_pixel_coords = create_homog_pixel_coords(B, H, W, depth.type())
    
    # Rays shooting out from target frame
    rays = multiply_coords(Kinv, homog_pixel_coords)
    
    # Points hitting objects out in the world
    world_points = depth * rays

    return world_points

def reconstruct_image(img, depth, pose, K):

    T = pose_vec2mat(pose)

    # Calculate 3D points in camera frame from depth map
    homog_world_points = to_homog_coords(depth_to_3d_points(depth, K))

    # The world points projected back into the reference view
    KT = pad_zero_column_right(K) @ to_homog_matrix(T)
    homog_projected_pixel_coords = multiply_coords(KT, homog_world_points)

    # Normalize homogeneous pixel coordinates
    projected_pixel_coords = from_homog_coords(homog_projected_pixel_coords)

    # Sample the source image in the projected pixel coordinates
    sampling_grid = to_sampling_grid(projected_pixel_coords)

    reconstruction = F.grid_sample(img, sampling_grid, padding_mode="border", align_corners=True)

    # Sampling points with abs value smaller than 1 are inside the frame
    valid_mask = sampling_grid.abs().max(dim=-1)[0] <=1

    debug = { 
        "homog_world_points": homog_world_points, 
        "projected_pixel_coords": projected_pixel_coords,
        "sampling_grid": sampling_grid }

    return reconstruction , valid_mask, debug
