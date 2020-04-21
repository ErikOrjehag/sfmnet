
import torch
import utils
import numpy as np

def eval_depth(gt, pred, crop=(0, 0)):
    assert gt.shape == pred.shape
    pred = pred.detach()
    mask = gt > 0 # Remove pixels missing from sparse ground truth
    crop_mask = torch.zeros_like(mask)
    H, W = crop_mask.shape[2:]
    crop_mask[:, :, crop[0]:H-crop[0], crop[1]:W-crop[1]] = 1
    mask = mask * crop_mask
    gt = gt[mask]
    pred = pred[mask]
    pred *= torch.median(gt) / torch.median(pred)
    pred = torch.clamp(pred, min=1e-3, max=80)
    metrics = compute_depth_val_metrics(gt, pred)
    return metrics

def compute_depth_val_metrics(gt, pred):

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    metrics = {
        "abs_rel": abs_rel, 
        "sq_rel": sq_rel, 
        "rmse": rmse, 
        "rmse_log": rmse_log, 
        "a1": a1, 
        "a2": a2, 
        "a3": a3 }

    metrics = utils.dict_tensors_to_num(metrics)

    return metrics

def eval_path(gt_poses, pred_poses):
    pred_poses = np.array(pred_poses)
    gt_poses = np.array(gt_poses)
    num_frames = len(gt_poses)
    track_length = 5
    ates = []
    for i in range(0, num_frames-1):
        pred_xyz = np.array(dump_xyz(pred_poses[i:i+track_length-1]))
        gt_xyz = np.array(dump_xyz(gt_poses[i:i+track_length-1]))
        ates.append(compute_ate(gt_xyz, pred_xyz))
    return ates

def dump_xyz(source_to_target_transforms):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3,3])
    for source_to_target_transform in source_to_target_transforms:
        cam_to_world = np.dot(cam_to_world, source_to_target_transform)
        xyzs.append(cam_to_world[:3,3])
    return xyzs

def compute_ate(gtruth_xyz, pred_xyz):
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse