
import torch
import data
from networks.deepconsensus import HomographyConsensus, HomographyConsensusLoss
from base_trainer import BaseTrainer

class HomographyTrainer(BaseTrainer):

    def __init__(self, args):
        super().__init__(
            args,
            loaders=data.get_batch_loader_split(args),
            model=HomographyConsensus(),
            loss_fn=HomographyConsensusLoss(pred=False)
        )

    #def get_parameter_groups(self):
    #    return [
    #        #{ 'params': self.model.siamese_unsuperpoint.parameters() },
    #        { 'params': self.model.pointnet_binseg.parameters() },
    #    ]

    
    def load_checkpoint(self, args):

        if args.load_consensus:
            model_dict = self.model.state_dict()
            consensus_checkpoint = torch.load(args.load_consensus, map_location=torch.device(args.device))
            model_dict.update(consensus_checkpoint["model"])
            self.model.load_state_dict(model_dict)

        if args.load_point:
            point_checkpoint = torch.load(args.load_point, map_location=torch.device(args.device))
            self.model.siamese_unsuperpoint.load_state_dict(point_checkpoint["model"])
        else:
            print("NEED TO LOAD siamese_unsuperpoint model!")
            exit()
        
        
        #self.optimizer.add_param_group({'params': self.model.siamese_unsuperpoint.parameters() })
        #self.optimizer.load_state_dict(point_checkpoint["optimizer"])
        #self.optimizer.add_param_group({'params': self.model.pointnet_binseg.parameters() })

        #state = self.optimizer.state_dict()

        #point_optim = point_checkpoint["optimizer"]
        #state['param_groups'][0].update(point_optim['param_groups'][0])
        #state['state'].update(point_optim['state'])
        
        #self.optimizer.load_state_dict(state)