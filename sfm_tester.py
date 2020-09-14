
import torch
import sfm_loss
import data
import networks.architectures
import utils
import numpy as np
import reconstruction
import geometry
import evaluation

def no_loss(data):
    return torch.tensor(0.0), {}

class SFMTester():

    def __init__(self, args):
        super().__init__()

        self.DEVICE = args.device

        self.loader = data.get_batch_loader_split(args)["test"]

        print("Test images: ", len(self.loader))

        self.model = networks.architectures.get_net(args)

        self.loss_fn = no_loss

        checkpoint = torch.load(args.load, map_location=torch.device(args.device))
        self.model.load_state_dict(checkpoint["model"])

        self.ates = []
        self.metrics = {}

        self.gt_poses = []
        self.pred_poses = []

        self.prev_tgt_i = 0

    def run(self):

        self.model.eval()

        utils.iterate_loader(self.DEVICE, self.loader, self._step_fn, end=len(self.loader))

        mean_ates = np.mean(self.ates)
        std_ates = np.std(self.ates)
        mean_metrics = utils.dict_mean(self.metrics)
        std_metrics = utils.dict_std(self.metrics)

        print("\n"+"="*20)
        print(f"Trajectory error: {mean_ates:0.3f}, std: {std_ates:0.3f}")
        print("Depth metrics:")
        for key in self.metrics:
            print(f"{key} -> mean: {mean_metrics[key]:0.3f}, std: {std_metrics[key]:0.3f}")
        print("="*20)

    def _step_fn(self, step, inputs):

        tgt_i = inputs['tgt_i'].item()
        
        # Forward pass and loss
        with torch.no_grad():
            loss, data = utils.forward_pass(self.model, self.loss_fn, inputs)

        # New sequence
        if self.prev_tgt_i != tgt_i - 1:
            print("\n"+"="*20+"\nNew sequence\n"+"="*20+"\n")
            self.ates += evaluation.eval_path(self.gt_poses, self.pred_poses)
            self.gt_poses = []
            self.pred_poses = []
        
        print(f"{step}/{len(self.loader)-1} - {tgt_i}")

        # Always
        poses = data["pose"]
        T_pred = utils.torch_to_numpy(geometry.to_homog_matrix(geometry.pose_vec2mat(poses[:,0])).squeeze(0))
        self.pred_poses.append(T_pred)
        T_gt = utils.torch_to_numpy(data["T"].squeeze(0))[1]
        self.gt_poses.append(T_gt)

        gt_depth = data["gt_sparse"]
        pred_depth = data["depth"][0]
        metrics = evaluation.eval_depth(gt_depth, pred_depth)

        self.metrics = utils.dict_append(self.metrics, metrics)

        self.prev_tgt_i = tgt_i