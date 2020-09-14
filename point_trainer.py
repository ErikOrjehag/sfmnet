
import data
from networks.unsuperpoint import SiameseUnsuperPoint, UnsuperLoss
from base_trainer import BaseTrainer

class PointTrainer(BaseTrainer):

    def __init__(self, args):
        super().__init__(
            args,
            loaders=data.get_batch_loader_split(args),
            model=SiameseUnsuperPoint(),
            loss_fn=UnsuperLoss()
        )