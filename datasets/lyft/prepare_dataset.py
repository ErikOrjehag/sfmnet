
import sys
sys.path.append("/home/ai/Code/sfmnet")

from pathlib import Path
from lyft import Lyft
import utils
import numpy as np
from imageio import imwrite
import viz

raw_path = sys.argv[1]
export_path = sys.argv[2]

print(f"raw: {raw_path}")
print(f"export: {export_path}")

dataset = Lyft(raw_path)
N = len(dataset)

prev_tgt_i = None
seq_i = -1
poses = []

for i, data in enumerate(dataset):
    tgt_img = np.uint8(utils.torch_to_numpy(data["tgt"]).transpose((1,2,0)) * 255)
    tgt_T = utils.torch_to_numpy(data["tgt_T"])
    T = utils.torch_to_numpy(data["T"])
    K = utils.torch_to_numpy(data["K"])
    tgt_T = utils.torch_to_numpy(data["tgt_T"])
    sparse = utils.torch_to_numpy(data["gt_sparse"].squeeze(0))
    tgt_i = data["tgt_i"]
    
    if prev_tgt_i != tgt_i - 1:
        if poses:
            poses = np.array([pose[:3,:].flatten() for pose in poses])
            p = f"{export_path}/seq{seq_i:010}"
            np.savetxt(f"{p}/poses.txt", poses)
            poses = []
        seq_i += 1
        print(f"new sequence: {seq_i:010}")
        p = f"{export_path}/seq{seq_i:010}"
        Path(p).mkdir(parents=True, exist_ok=True)    
        np.savetxt(f"{p}/cam.txt", K)
    prev_tgt_i = tgt_i
    
    t = T[0,:3,3]
    l = np.linalg.norm(t)

    if l < 0.3:
        print(f"---ignore---> {i}/{N}: {l}")
    else:
        f = f"{p}/img{tgt_i:010}.jpg"
        f2 = f"{p}/img{tgt_i:010}.npy"
        print(f"---save---> {i}/{N}: {l} - {f}")
        poses.append(tgt_T)
        imwrite(f, tgt_img)
        np.save(f2, sparse)
