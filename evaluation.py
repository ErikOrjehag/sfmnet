
import torch
import utils

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