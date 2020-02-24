
import data
from networks.unsuperpoint import UnsuperPoint, UnsuperLoss

class PointTrainer():

    def __init__(self, args):
        super().__init__(
            loaders=data.get_coco_batch_loader_split(args),
            model=UnsuperPoint()
            loss_fn=UnsuperLoss()
        )