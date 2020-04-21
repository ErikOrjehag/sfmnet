
import data
import sfm_loss
import networks.architectures
import evaluation
from base_trainer import BaseTrainer

class SfMTrainer(BaseTrainer):

    def __init__(self, args):
        super().__init__(
            args,
            loaders=data.get_batch_loader_split(args),
            model=networks.architectures.get_net(args),
            loss_fn=sfm_loss.get_loss_fn(args)
        )

    def calc_metrics(self, data):
        gt_depth = data["gt_sparse"]
        pred_depth = data["depth"][0]
        metrics = evaluation.eval_depth(gt_depth, pred_depth)
        return metrics