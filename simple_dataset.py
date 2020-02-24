
import os
import glob
import torch
import torch.utils.data as data
from imageio import imread
import utils

class SimpleDataset(data.Dataset):

    def __init__(self, root):
        super().__init__()

        self.images = sorted(glob.glob(os.path.join(root, "*.jpg")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = imread(self.images[index])
        img = utils.cv2_to_torch(img)
        data = { 
            "img": torch.tensor(img)
        }
        return data