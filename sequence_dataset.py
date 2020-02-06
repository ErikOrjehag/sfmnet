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
  def __init__(self, root):
    super().__init__()

    # Sequence folders
    self.sequences = sorted(glob.glob(os.path.join(root, "*", "")))
    
    # Camera intrinsics
    self.intrinsics = [
      np.genfromtxt(os.path.join(sequence, "cam.txt"), 
        delimiter=" ", dtype=np.float32).reshape((3, 3))
      for sequence in self.sequences]
    self.inv_intrinsics = [np.linalg.inv(intrinsics) for intrinsics in self.intrinsics]
    
    # Poses
    
    self.poses = []
    for seq in self.sequences:
      with open(os.path.join(seq, "poses.txt")) as f:
        p = []
        lines = f.readlines()
        for line in lines:
          p.append(np.fromstring(line, sep=" ", dtype=np.float32).reshape(3, 4))
        self.poses.append(p)
    
    # Images
    self.images = [
      sorted(glob.glob(os.path.join(sequence, "*.jpg"))) 
      for sequence in self.sequences]
    
    # Length helpers
    self.lens = [len(images) - 2 for images in self.images]
    self.len = sum(self.lens)
    self.cumlen = np.hstack((0, np.cumsum(self.lens)))

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
        sparse = np.load(paths[0][:-4] + ".npy").astype(np.float32)
        dense = np.load(paths[0][:-4] + "_dense.npy").astype(np.float32)
        K = self.intrinsics[i]
        Kinv = self.inv_intrinsics[i]
        tgt_pose = self.poses[i][tgt]
        ref_pose = [self.poses[i][tgt-1], self.poses[i][tgt+1]]
        return [torch.tensor(thing) for thing in [tgt_img, ref_imgs, K, Kinv, sparse, dense, tgt_pose, ref_pose]]

if __name__ == '__main__':
  dataset = SequenceDataset(sys.argv[1])
  for i in range(0, len(dataset)):
    tgt, refs, K, Kinv, sparse, dense, tgt_pose, ref_pose = dataset[i][:8]
    print(relative_transform(tgt_pose, ref_pose[0]))
    print(relative_transform(tgt_pose, ref_pose[1]))
    print("---")
    img = torch.cat((refs[0], tgt, refs[1]), dim=1)
    cv2.imshow("img", viz.tensor2img(img))
    cv2.imshow("sparse", viz.tensor2depthimg(sparse))
    cv2.imshow("dense", viz.tensor2depthimg(dense))
    cv2.waitKey(0)