
import torch
import data
from networks.deepconsensus import FundamentalConsensus, FundamentalConsensusLoss
from base_trainer import BaseTrainer

class FundamentalTrainer(BaseTrainer):

    def __init__(self, args):
        super().__init__(
            args,
            loaders=data.get_coco_batch_loader_split(args),
            model=FundamentalConsensus(),
            loss_fn=FundamentalConsensusLoss()
        )

    def load_checkpoint(self, args):
        point_checkpoint = torch.load(args.load_point, map_location=torch.device(args.device))
        self.model.siamese_unsuperpoint.load_state_dict(point_checkpoint["model"])