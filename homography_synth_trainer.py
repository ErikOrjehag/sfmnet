
import torch
import data
from networks.deepconsensus import HomographyConsensusSynthPoints, HomographyConsensusSynthPointsLoss, HomographyConsensusLoss
from base_trainer import BaseTrainer
from homo_adap_dataset import HomoAdapSynthPointDataset

class HomographySynthTrainer(BaseTrainer):

    def __init__(self, args):
        super().__init__(
            args,
            loaders=data.get_batch_loader_split(args),
            model=HomographyConsensusSynthPoints(),
            loss_fn=HomographyConsensusLoss(pred=False)
        )
    
    def load_checkpoint(self, args):
        pass