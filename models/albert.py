from utils import *
import torch
from transformers import AlbertModel, AlbertConfig, BertTokenizerFast
import copy

class TrainConfig(TrainConfigBase):
    random_seed : int = 1
    # pretrained_path : str = 'ckiplab/albert-base-chinese' # pretrained model path or Huggingface model name
    pretrained_path : str = 'models_pretrained/albert-base-chinese' # pretrained model path or Huggingface model name
    model_name : str = 'albert'
    start_saving_epoch : int = 5
    num_epoches : int = 6
    batch_size : int = 64
    eval_batch_size : int = 1024
    test_batch_size : int = 1024
    eval_by_steps : int = 400
    dataset_cache_size : int = 180000
    min_lr = 1e-9
    max_lr = 6e-5
    warmup_epochs = 1

    def __init__(self):
        self.classes=[x.strip() for x in open(self.data_path_class).readlines()]
        self.num_classes=len(self.classes)
        self.model_tokenizer=BertTokenizerFast.from_pretrained(self.pretrained_path)
        self.model_config=AlbertConfig.from_pretrained(self.pretrained_path)

    def create_optimizer(self, model: torch.nn.Module):
        model.train()
        param_optimizer=list(model.named_parameters())
        no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
        optimizer_grouped_parameters=[
            {
                'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay':0.06
            },
            {
                'params':[p for n ,p in param_optimizer if any(nd in n for nd in no_decay) ],
                'weight_decay':0.0
            }
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.max_lr)
        return self.optimizer
    
class Model(ModelBase):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        config =  train_config.model_config
        hd_size = config.hidden_size
        self.train_config = train_config
        self.albert = AlbertModel.from_pretrained(train_config.pretrained_path)

        self.classification = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hd_size, hd_size),
            torch.nn.BatchNorm1d(hd_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hd_size, train_config.num_classes),
        )

    def forward(self, x):
        o = self.albert(**x)
        x = self.classification(o.pooler_output + o.last_hidden_state[:, 0, :])
        return x

    def collate_fn(self, batch : list):
        prefix = [self.train_config.model_tokenizer.cls_token_id]
        suffix = [self.train_config.model_tokenizer.sep_token_id]

        tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(prefix + data['x'] + suffix) for data in batch], 
            batch_first=True, 
            padding_value=0).to(self.train_config.device)
        
        x = { 'input_ids': tokens, 'attention_mask': (tokens != 0).float() } # bert forward 参数
        y = torch.tensor([data['y'] for data in batch], device=self.train_config.device)
        return x, y
    
class TrainScheduler(TrainSchedulerBase):
    model : Model
    train_config : TrainConfig
    
    def on_start(self, epoch_steps: int):
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.train_config.optimizer,
            start_factor = self.train_config.min_lr / self.train_config.max_lr,
            total_iters = epoch_steps * self.train_config.warmup_epochs
        )
        self.cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.train_config.optimizer, 
            T_max = epoch_steps * (self.train_config.num_epoches - self.train_config.warmup_epochs)
        )
    
    def on_step_end(self, epoch : int, step: int, t_loss: float, t_acc : float):
        if epoch < self.train_config.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            self.cos_scheduler.step()