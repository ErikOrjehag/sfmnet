import torch
import torch.nn.functional as F

def pose_vec2mat(vec):
    t = vec[:,:3].unsqueeze(-1) # [B, 3, 1]
    r = vec[:,3:]
    R = euler2mat(r) # [B, 3, 3]
    transform = torch.cat([R, t], dim=2) # [B, 3, 4]
    return transform

def to_homog_coords(coords):
    pad = (0, 0, 0, 0, 0, 1) # [B,C,H,W]
    if len(coords.shape) == 3: # [B,C,N]
        pad = pad[2:]
    return F.pad(input=coords, pad=pad, mode="constant", value=1)

def to_homog_matrix(matrix):
    matrix = F.pad(input=matrix, pad=(0, 0, 0, 1), mode="constant", value=0)
    matrix[...,-1,-1] = 1.0
    return matrix

def from_homog_coords(coords):
    X = coords[:,0]
    Y = coords[:,1]
    Z = coords[:,2].clamp(min=1e-3)
    return torch.stack((X/Z, Y/Z), dim=1)

def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat