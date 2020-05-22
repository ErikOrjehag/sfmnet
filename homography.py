
import numpy as np
from numpy.random import uniform, randint
from scipy.stats import truncnorm
import torch
import torch.nn.functional as F
import cv2
import viz

def random_euclidian(rotation, translation, scale):
    r = np.random.uniform(-1, 1) * rotation
    s = np.random.uniform(-1, 1) * scale + 1
    tx = np.random.uniform(-1, 1) * translation
    ty = np.random.uniform(-1, 1) * translation
    T = np.array([
        [ np.cos(r)*s, -np.sin(r)  , tx],
        [ np.sin(r)  ,  np.cos(r)*s, ty],
        [ 0, 0, 1 ],
    ])
    return torch.tensor(T, dtype=torch.float32)

def random_sheer(sheer):
    s = np.random.uniform(-1, 1) * sheer
    T = np.array([
        [ 1,  s, 0 ],
        [ s,  1, 0 ],
        [ 0,  0, 1 ],
    ])
    return torch.tensor(T, dtype=torch.float32)

def random_projective(projective):
    p = np.random.uniform(-1, 1) * projective
    T = np.array([
        [ 1,  0, 0 ],
        [ 0,  1, 0 ],
        [ p,  p, 1 ],
    ])
    return torch.tensor(T, dtype=torch.float32)

def random_homography(rotation=0, translation=0, scale=0, sheer=0, projective=0):
    He = random_euclidian(rotation, translation, scale)
    Hs = random_sheer(sheer)
    Hp = random_projective(projective)
    return He @ Hs @ Hp

def random_homographies(batch, **args):
    return torch.stack([ random_homography(**args) for _ in range(batch) ])

def homo_warp_points(points, homography):
    """
    points: [N,2]
    homography: [B,H,W]
    """
    B = homography.shape[0]
    points = torch.cat((
        points, 
        torch.ones((points.shape[0], 1))), dim=1)
    warped_points = (homography@points.transpose(0, 1)).transpose(2, 1)
    warped_points = warped_points[:,:,:2] / warped_points[:,:,2:]
    return warped_points

def homo_warp_grid(grid, homography):
    """
    grid: [H,W,2]
    homography: [B,3,3]
    """
    B, H, W = homography.shape[0], *grid.shape[:2]
    points = grid.view(-1, 2)
    warped_points = homo_warp_points(points, homography)
    warped_grid = warped_points.view(B,H,W,2)
    return warped_grid

def warp_image(img, homography):
    """
    img: [B?,C,H,W]
    homography: [B?,3,3]
    """
    has_batch = len(img.shape) == 4
    if not has_batch:
        img = img.unsqueeze(0)
        homography = homography.unsqueeze(0)
    B, C, H, W = img.shape
    r = H/W
    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(-W/2,W/2,W),
            torch.linspace(-H/2,H/2,H)), dim=2) \
        .transpose(0,1) \
        .to(img.device) \
        .contiguous()
    warped_grid = homo_warp_grid(grid, torch.inverse(homography))
    warped_grid[:,:,:,0] /= (W/2)
    warped_grid[:,:,:,1] /= (H/2)
    warped_img = F.grid_sample(img, warped_grid, mode="bilinear", align_corners=True)
    if not has_batch:
        warped_img = warped_img.squeeze(0)
    return warped_img
    

if __name__ == "__main__":
    img = torch.tensor(np.random.rand(4,3,320,460), dtype=torch.float32)
    
    while True:
        
        homography = random_homographies(img.shape[0], rotation=np.pi/8, translation=50, scale=0.1, sheer=0.1, projective=0.001)
        
        warp = warp_image(img, homography)

        img_stack = torch.cat((*img,), dim=2)
        warp_stack = torch.cat((*warp,), dim=2)

        cv2.imshow("img", viz.tensor2img(img_stack))
        cv2.imshow("warp", viz.tensor2img(warp_stack))
        
        key = cv2.waitKey(0)
        if key == 27:
            break