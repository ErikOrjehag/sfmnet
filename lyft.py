
import os
from sequence_dataset import SequenceDataset
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
import numpy as np
from pyquaternion import Quaternion

class Lyft(SequenceDataset):

    def __init__(self, root, split=None):

        self.data = LyftDataset(
            data_path=root, 
            json_path=os.path.join(root, os.path.basename(os.path.normpath(root))), 
            verbose=False)

        self.explorer = LyftDatasetExplorer(self.data)
        
        super().__init__(root, split)
        
    def _get_sequences(self):
        return [(scene["first_sample_token"], "CAM_FRONT") for scene in self.data.scene]

    def _get_samples(self, sequence):
        FIRST_TOKEN, CAMERA = sequence
        next_token = FIRST_TOKEN
        samples = []
        while next_token != "":
            sample = self.data.get("sample", next_token)
            next_token = sample["next"]
            image = sample["data"][CAMERA]
            samples.append(image)
        return samples

    def _load_sample(self, seq_i, sample):
        
        sd_record = self.data.get("sample_data", sample)
        cs_record = self.data.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        pose_record = self.data.get("ego_pose", sd_record["ego_pose_token"])
        
        K = np.array(cs_record["camera_intrinsic"]).astype(np.float32)
        
        T = Quaternion(pose_record["rotation"]).transformation_matrix
        T[:3,3] = np.array(pose_record["translation"])
        #print("before", T)
        T = T @ np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
        #print("after", T)
        
        
        img, K = self._load_image_from_disk(self.data.get_sample_data_path(sample), K)

        return [img, T, K]
        
    def _load_depth(self, sample):
        sample_data = self.data.get("sample_data", sample)
        LIDAR_TOKEN = self.data.get("sample", sample_data["sample_token"])["data"]["LIDAR_TOP"]
        points, depths, image = self.explorer.map_pointcloud_to_image(LIDAR_TOKEN, sample)
        W, H = image.size
        crop, scale = self.calc_crop(H, W)
        dmap = np.zeros(self.imHW, dtype=np.float64)
        coords = points[:2,:] * scale
        coords[1] -= crop
        mask = np.logical_and(coords[1,:] >= 0, coords[1,:] < self.imHW[0])
        coords = coords[:,mask].astype(np.int)
        depths = depths[mask]
        dmap[coords[1], coords[0]] = depths
        return dmap
