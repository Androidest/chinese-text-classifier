from .TrainConfigBase import TrainConfigBase
import torch

class TrainSchedulerBase:
    def __init__(self, train_config: TrainConfigBase, model: torch.nn.Module):
        self.train_config = train_config
        self.model = model

    # preprocess data from the dataset, 
    # the output structure is directly used for model training, 
    # and x will be directly passed into the forward function of the model
    def on_collate(self, batch : list):
        pass

    # on start training
    def on_start(self):
        pass

    # on end of a batch
    def on_step_end(self, step: int, t_loss: float):
        pass