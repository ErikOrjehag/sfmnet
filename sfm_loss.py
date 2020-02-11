import torch
import torch.nn.functional as F
from reconstruction import reconstruct_image
import utils

def get_loss_fn(args):
    return SfmLoss(
        weights={
            "photo": 1.0, 
            "smooth": args.smooth_weight, 
            "explain": args.explain_weight },
        ssim_weight=args.ssim_weight,
        which_smooth_map=args.which_smooth_map,
        use_normalization=args.smooth_map_normalization,
        use_edge_aware=args.edge_aware,
        use_upscale=args.upscale,
        use_stationary_mask=args.stationary_mask,
        use_min=args.min_loss)

class SfmLoss():

    def __init__(self, 
        weights, 
        which_smooth_map, 
        ssim_weight,
        use_normalization, 
        use_edge_aware, 
        use_upscale, 
        use_stationary_mask,
        use_min):
        """
        weights - Used to weigh the different loss terms against each other.
        which_smooth_map - What map to use in the smoothness term, either "disp" or "depth".
        ssim_weight - Used to balance L1 and SSIM terms, can be zero to disable SSIM.
        use_normalization - Should the disp/depth map be normalized before the smoothness term is calculated?
        use_edge_aware - Should the edge aware or second order smoothness term be used?
        use_upscale - Should the disp/depth maps be upscaled to the original image size?
        use_ssim - Should the SSIM term be used in the photometric loss?
        use_stationary_mask - Mask out stationary pixels.
        use_min - Use average photo loss over all reference images, else use minimum
        """
        self.weights = weights
        self.which_smooth_map = which_smooth_map
        self.ssim_weight = ssim_weight
        self.use_normalization = use_normalization
        self.use_edge_aware = use_edge_aware
        self.use_upscale = use_upscale
        self.use_stationary_mask = use_stationary_mask
        self.use_min = use_min

    def __call__(self, data):

        total_loss = 0.0

        # Photometric loss
        photo_loss, debug = photometric_reconstruction_loss(
            data, 
            ssim_weight=self.ssim_weight,
            use_upscale=self.use_upscale, 
            use_stationary_mask=self.use_stationary_mask,
            use_min=self.use_min)
        total_loss += self.weights["photo"] * photo_loss
        
        # Smooth loss
        smooth_map = data[self.which_smooth_map]
        if self.use_normalization:
            smooth_map = utils.normalize_map(smooth_map)
        if self.use_edge_aware:
            smooth_loss = edge_aware_smooth_loss(smooth_map)
        else:
            smooth_loss = second_order_smooth_loss(smooth_map)
        total_loss += self.weights["smooth"] * smooth_loss

        # Explainability regularization
        if "exp_mask" in data:
            explain_loss = explainability_regularization_loss(data["exp_mask"])
            total_loss += self.weights["explain"] * explain_loss

        return total_loss, debug

def photometric_reconstruction_loss(
    data, 
    ssim_weight, 
    use_upscale, 
    use_stationary_mask, 
    use_min):
    
    depth = data["depth"]
    exp_mask = data["exp_mask"] if "exp_mask" in data else [None] * len(depth)
    assert len(depth) == len(exp_mask)

    total_loss = 0.0
    total_debug = {}

    # For every scale in the pyramid
    for depth, exp_mask in zip(depth, exp_mask):
        tgt = data["tgt"]
        refs = data["refs"]
        K = data["K"]
        if use_upscale:
            # Upscale depth map to input image size
            H, W = tgt.shape[2:]
            depth = F.interpolate(depth, (H, W), mode="area")
            if exp_mask is not None:
                exp_mask = F.interpolate(exp_mask, (H, W), mode="bilinear")
        else:
            # Downscale target and reference images to depth map size
            H, W = depth.shape[2:]
            ratio = tgt.shape[2] / H
            tgt = F.interpolate(tgt, (H, W), mode="area")
            refs = F.interpolate(refs, (H, W), mode="area")
            K = torch.cat((K[:,:2] / downscale, K[:,2:]), dim=1)
        # Calculate the photometric reconstruction loss for a single scale
        loss, debug = one_scale_photometric_loss(
            depth=depth, 
            tgt=tgt, 
            refs=refs, 
            K=K, 
            exp_map=exp_mask, 
            ssim_weight=ssim_weight, 
            use_stationary_mask=use_stationary_mask, 
            use_min=use_min)
        # Add to total
        total_loss += loss
        for k, v in debug.items():
            if k not in total_debug:
                total_debug[k] = []
            total_debug[k].append(v)

    return total_loss, total_debug

def one_scale_photometric_loss(
    depth, 
    tgt, 
    refs, 
    K, 
    exp_mask, 
    ssim_weight,
    use_stationary_mask, 
    use_min):

    total_loss = 0.0
    
    warps = []
    reconstruction_similarities = []
    stationary_similarities = []

    for i, ref in enumerate(refs.split(split_size=1, dim=1)):
        pose = data["pose"][:,i]

        ref_warped, inside_mask = reconstruct_image(ref, depth, pose, K)[:2]
        reconstruction_similarity = photometric_similarity_map(tgt, ref_warped, ssim_weight)
        reconstruction_similarity *= inside_mask.unsqueeze(1).float()

        warps.append(ref_warped)
        
        if exp_mask is not None:
            reconstruction_similarity *= exp_mask[:,i].unsqueeze(1)

        if use_stationary_mask:
            stationary_similarity = photometric_similarity_map(tgt, ref, ssim_weight)
            stationary_similarity += utils.randn_like(stationary_similarity) * 1e-5 # break ties (not needed when using not_stationary_mask???)
    
    reconstruction_similarities = torch.cat(reconstruction_similarities, dim=1)
    stationary_similarities = torch.cat(stationary_similarities, dim=1)

    if not use_min:
        reconstruction_similarities = reconstruction_similarities.mean(dim=1, keepdim=True)

    combined = torch.cat((stationary_similarities, reconstruction_similarities), dim=1)

    min_similarities, min_idx = torch.min(combined, dim=1)
    not_stationary_mask = (min_idx > stationary_similarities.shape[1] - 1).float()

    total_loss = min_similarities[not_stationary_mask].mean()

    return total_loss, { "warp": warps, "diff": to_optimize, "min_idx": min_idx }


def photometric_similarity_map(img1, img2, ssim_weight):
    diff = (img1 - img2).abs()
    L1 = diff.mean(dim=1, keepdim=True)
    if ssim_weight != 0:
        ssim = calculate_ssim(img1, img2).mean(dim=1, keepdim=True)
        similarity_map = ssim_weight * ssim + (1 - ssim_weight) * L1
    else:
        similarity_map = L1
    return similarity_map

def calculate_ssim(x, y):
    refl = lambda z: F.pad(z, pad=(1,1,1,1), mode="reflect")
    pool = lambda z: F.avg_pool2d(z, kernel_size=3, stride=1)
    x = refl(x)
    y = refl(y)
    mu_x = pool(x)
    mu_y = pool(y)
    sigma_x = pool(x ** 2) - mu_x ** 2
    sigma_y = pool(y ** 2) - mu_y ** 2
    sigma_xy = pool(x * y) - mu_x * mu_y
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return torch.clamp((1 - ssim_n / ssim_d) / 2, min=0, max=1)

def second_order_smooth_loss(depths):
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

def edge_aware_smooth_loss(disps, img):
    loss = 0.0
    for scale, disp in enumerate(disps):
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)
        loss += (grad_disp_x.mean() + grad_disp_y.mean()) / (2 ** scale)
    return loss

def explainability_regularization_loss(masks):
    loss = 0
    for i, mask in enumerate(masks):
        ones = torch.ones_like(mask)
        loss += F.binary_cross_entropy(mask, ones)
    return loss