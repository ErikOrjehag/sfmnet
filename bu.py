import torch.utils.data as data
import torchvision
import numpy as np
from imageio import imread
import cv2
import glob
import os
import sys
import viz
import torch

def relative_transform(from_T, to_T):
  R1, t1 = from_T[:3,:3], from_T[:3,3:]
  R2, t2 = to_T[:3,:3], to_T[:3,3:]
  rel_T = np.vstack((np.hstack((R2.T @ R1, R2.T @ (t1 - t2))), (0,0,0,1)))
  return rel_T

class SequenceDataset(data.Dataset):
  def __init__(self, root, load_gt=True):
    super().__init__()

    self.load_gt = load_gt

    # Sequence folders
    sequences = sorted(glob.glob(os.path.join(root, "*", "")))
    
    # Camera intrinsics
    self.intrinsics = [
      np.genfromtxt(os.path.join(sequence, "cam.txt"), 
        delimiter=" ", dtype=np.float32).reshape((3, 3))
      for sequence in sequences]
    self.inv_intrinsics = [np.linalg.inv(intrinsics) for intrinsics in self.intrinsics]
    
    # Images
    self.images = [
      sorted(glob.glob(os.path.join(sequence, "*.jpg"))) 
      for sequence in sequences]
    
    # Length helpers
    self.lens = [len(images) - 2 for images in self.images]
    self.len = sum(self.lens)
    self.cumlen = np.hstack((0, np.cumsum(self.lens)))

    if self.load_gt:
      # Poses
      self.poses = []
      for seq in sequences:
        with open(os.path.join(seq, "poses.txt")) as f:
          p = []
          lines = f.readlines()
          for line in lines:
            p.append(np.fromstring(line, sep=" ", dtype=np.float32).reshape(3, 4))
          self.poses.append(p)

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    for i in range(len(self.cumlen)):
      if index - self.cumlen[i] < 0:
        i -= 1
        tgt = 1 + index - self.cumlen[i]
        seq = self.images[i]
        paths = [seq[tgt], seq[tgt-1], seq[tgt+1]]
        imgs = np.transpose([(np.array(imread(path)).astype(np.float32)/255)*2-1 for path in paths], (0, 3, 1, 2))
        tgt_img = imgs[0]
        ref_imgs = imgs[1:]
        K = self.intrinsics[i]
        Kinv = self.inv_intrinsics[i]
        data = {
          "tgt": torch.tensor(tgt_img),
          "refs": torch.tensor(ref_imgs),
          "K": torch.tensor(K),
        }
        if self.load_gt:
          sparse = np.load(paths[0][:-4] + ".npy").astype(np.float32)
          dense = np.load(paths[0][:-4] + "_dense.npy").astype(np.float32)
          tgt_pose = self.poses[i][tgt]
          ref_pose = [self.poses[i][tgt-1], self.poses[i][tgt+1]]
          data = {
            **data,
            "gt_sparse": torch.tensor(sparse).unsqueeze(0),
            "gt_dense": torch.tensor(dense).unsqueeze(0),
            "gt_pose": torch.tensor(0)
          }
        return data

if __name__ == '__main__':
  dataset = SequenceDataset(sys.argv[1])
  for i in range(0, len(dataset)):
    data = dataset[i]
    print(data["gt_sparse"].shape)
    
    #print(relative_transform(tgt_pose, ref_pose[0]))
    #print(relative_transform(tgt_pose, ref_pose[1]))
    #print("---")
    img = torch.cat((data["refs"][0], data["tgt"], data["refs"][1]), dim=1)
    cv2.imshow("img", viz.tensor2img(img))
    cv2.imshow("gt_sparse", viz.tensor2depthimg(data["gt_sparse"]))
    cv2.imshow("gt_dense", viz.tensor2depthimg(data["gt_dense"]))
    
    key = cv2.waitKey(0)
    if key == 27:
      break