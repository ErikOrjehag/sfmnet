
import glob
import os
import numpy as np
from sequence_dataset import SequenceDataset
import json

def Rt_to_T(Rt):
    return np.vstack((Rt, np.array([0, 0, 0, 1])))

def get_intrinsics(sequence):
    p = sorted(glob.glob(os.path.join(sequence, "Information", "*.json")))[0]
    print(p)
    with open(p) as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsic"]["matrix"]).reshape(4,4)
    extrinsic = np.array(data["extrinsic"]["matrix"]).reshape(4,4)
    print(intrinsics)
    print(extrinsic)
    exit()
    #np.genfromtxt(os.path.join(sequence, "cam.txt"), 
    #        delimiter=" ", dtype=np.float32).reshape((3, 3))

class Synthia(SequenceDataset):

    def __init__(self, root, load_gt=True):
        super().__init__(root, load_gt)

        self.intrinsics = [
            get_intrinsics(sequence)
            for sequence in self.sequences]

        self.poses = []
        for sequence in self.sequences:
            with open(os.path.join(sequence, "poses.txt")) as f:
                self.poses.append( [ Rt_to_T(np.fromstring(line, sep=" ", dtype=np.float64).reshape(3, 4)) for line in f.readlines() ] )

    def _get_sequences(self):
        return sorted(glob.glob(os.path.join(self.root, "*", "*", "")))

    def _get_samples(self, sequence):
        return [(i, path) for i, path in enumerate(sorted(glob.glob(os.path.join(sequence, "*.jpg"))))]

    def _load_sample(self, seq_i, sample):
        i, path = sample
        K = self.intrinsics[seq_i]
        T = self.poses[seq_i][i]
        img, K = self._load_image_from_disk(path, K)
        return [img, T, K]

    def _load_depth(self, sample):
        i, path = sample
        return np.load(path[:-4] + ".npy").astype(np.float32)
        