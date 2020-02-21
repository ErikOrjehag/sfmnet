
import glob
import os
import numpy as np
from sequence_dataset import SequenceDataset

def Rt_to_T(Rt):
    return np.vstack((Rt, np.array([0, 0, 0, 1])))

class Kitti(SequenceDataset):

    def __init__(self, root, load_gt=True):
        super().__init__(root, load_gt)

        self.intrinsics = [
            np.genfromtxt(os.path.join(sequence, "cam.txt"), 
            delimiter=" ", dtype=np.float32).reshape((3, 3))
            for sequence in self.sequences]

        self.poses = []
        for sequence in self.sequences:
            with open(os.path.join(sequence, "poses.txt")) as f:
                self.poses.append( [ Rt_to_T(np.fromstring(line, sep=" ", dtype=np.float64).reshape(3, 4)) for line in f.readlines()] )

    def _get_sequences(self):
        return sorted(glob.glob(os.path.join(self.root, "*", "")))

    def _get_samples(self, sequence):
        return sorted(glob.glob(os.path.join(sequence, "*.jpg")))

    def _load_sample(self, seq_i, sample):
        img = self._load_image_from_disk(sample)
        K = self.intrinsics[seq_i]
        T = self.poses[seq_i]
        return img, T, K

    def _load_depth(self, sample):
        return np.load(sample[:-4] + ".npy").astype(np.float32)
        