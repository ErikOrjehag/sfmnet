import torch
import ignite
from ignite.metrics import Accuracy
import numpy as np
from sequence_dataset import SequenceDataset
import utils
import argparse
from sfm_loss import SfmLoss
from sfm_net import SfmNet

# Parse arguments
parser = argparse.ArgumentParser(description='Train and eval sfm nets.')
parser.add_argument('--batch', default=8, type=int, help='The batch size.')
parser.add_argument('--workers', default=24, type=int, help='The number of worker threads.')
parser.add_argument('--device', default="cuda", type=str, help='The device to run on cpu/cuda.')
args = parser.parse_args()
print('\nCurrent arguments -> ', args, '\n')

# Construct datasets
dataset = SequenceDataset("/home/ai/Data/kitti_formatted")
train_loader, val_loader, test_loader = utils.data_loaders(
  dataset, 0.7, 0.1, 0.2, args.batch, args.workers)

model = SfmNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = SfmLoss()

max_epochs = 10
validate_every = 1000
checkpoint_every = 1000

trainer = ignite.engine.create_supervised_trainer(model, optimizer, criterion, device=args.device)

evaluator = ignite.engine.create_supervised_evaluator(model, device=args.device,
  metrics={"loss": ignite.metrics.Loss(criterion)})

@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def validate(trainer):
  if trainer.state.iteration % validate_every == 0:
    print("Run evaluator")
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("After {} iterations, loss = {:2f}"
      .format(trainer.state.iteration, metrics["loss"]))

checkpointer = ignite.handlers.ModelCheckpoint("./checkpoints", "my_model",
  save_interval=checkpoint_every, create_dir=True, require_empty=False)
trainer.add_event_handler(ignite.engine.Events.ITERATION_COMPLETED, checkpointer, {"model": model})

trainer.run(train_loader, max_epochs=max_epochs)