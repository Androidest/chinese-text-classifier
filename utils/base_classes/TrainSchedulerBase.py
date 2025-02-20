from .TrainConfigBase import TrainConfigBase
import torch

class TrainSchedulerBase:
    def __init__(self, train_config: TrainConfigBase, model: torch.nn.Module):
        self.train_config = train_config
        self.model = model
        
    # on start training
    def on_start(self):
        pass

    # on end of a batch
    def on_step_end(self, step: int, t_loss: float):
        pass