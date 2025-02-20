from .TrainConfigBase import TrainConfigBase
import torch

class ModelBase(torch.nn.Module):
    # preprocess data from the dataset, 
    # the output structure is directly used for model training, 
    # and x will be directly passed into the forward function of the model
    def collate_fn(self, batch : list):
        raise NotImplementedError