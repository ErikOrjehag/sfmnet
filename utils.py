import torch
import numpy as np
import random

def iterate_loader(device, loader, fn, start=0, end=None, args=[]):
    for step, inputs in enumerate(loader, start=start):
        if end is not None and step >= end:
            break
        inputs = dict_to_device(inputs, device)
        fn(step, inputs, *args)

def forward_pass(model, loss_fn, inputs):
    outputs = model(inputs)
    data = { **inputs, **outputs }
    loss, debug = loss_fn(data)
    data = { **data, **debug }
    return loss, data

def backward_pass(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def dict_to_device(keyval, device):
    return map_dict(keyval, lambda key, val: val.to(device))

def dict_tensors_to_num(keyval):
    return map_dict(keyval, lambda key, val: val.cpu().item())

def dict_append(d1, d2):
    for key, val in d2.items():
        if key not in d1:
            d1[key] = [ val ]
    def fun(key, val):
        if isinstance(val, list):
            return val + [ d2[key] ]
        else:
            return [ val ] + [ d2[key] ]
    return map_dict(d1, fun)

def dict_mean(d):
    return map_dict(d, lambda key, val: np.mean(val))

def dict_std(d):
    return map_dict(d, lambda key, val: np.std(val))

def map_dict(keyval, f):
    return { key: f(key, val) for key, val in keyval.items() }

def normalize_map(map):
    mean_map = map.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    norm_map = map / (mean_map + 1e-7)
    return norm_map

def randn_like(tensor):
    return torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device)

def sigmoid_to_disp_depth(sigmoid, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = min_disp + (max_disp - min_disp) * sigmoid
    depth = 1 / disp
    return disp, depth

def sec_to_hms(t):
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return "{:02d}h{:02d}m{:02d}s".format(t, m, s)

def sum_to_dict(target, source):
    for key, val in source.items():
        if key in target:
            target[key] += val
        else:
            target[key] = val

def is_interval(step, interval):
  return step != 0 and step % interval == 0

def normalize_image(img):
    #x = (input_image - 0.45) / 0.225
    # (img - 0.5) * 0.225 from unsuperpoint paper
    return img * 2 - 1

def cv2_to_torch(img):
    return torch.tensor(np.transpose(np.array(img).astype(np.float32) / 255, axes=(2, 0, 1)))

def torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()