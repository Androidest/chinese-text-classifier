import torch
from transformers import AlbertModel, AlbertConfig, BertTokenizerFast
from utils import TrainSchedulerBase, TrainConfigBase
import copy

class TrainConfig(TrainConfigBase):
    pretrained_path : str = 'ckiplab/albert-base-chinese'
    # pretrained_path : str = 'models_pretrained/albert-base-chinese'
    save_path : str = 'models_fine_tuned'
    model_name : str = 'albert'
    start_saving_epoch : int = 4
    num_epoches : int = 9
    batch_size : int = 128
    eval_batch_size : int = 64
    test_batch_size : int = 1024
    eval_by_steps : int = 200
    dataset_cache_size : int = 100000
    # Staged training: Unlock different parameters and change the learning rate at different stages
    stage_start_step : list = [0, 2800*2, 2800*4] # The step at which each stage starts
    stage_lr : list = [5e-5, 2e-5, 9e-6] # The learning rate at each stage

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
                'weight_decay':0.01
            },
            {
                'params':[p for n ,p in param_optimizer if any(nd in n for nd in no_decay) ],
                'weight_decay':0.0
            }
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        return self.optimizer

class Model(torch.nn.Module):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        config =  train_config.model_config
        hd_size = config.hidden_size
        self.train_config = train_config
        self.albert = AlbertModel.from_pretrained(train_config.pretrained_path)

        config.num_hidden_groups = 3
        layerGroup = self.albert.encoder.albert_layer_groups[0]
        for _ in range(1, config.num_hidden_groups):
            self.albert.encoder.albert_layer_groups.append(copy.deepcopy(layerGroup))

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

class TrainScheduler(TrainSchedulerBase):
    model : Model
    train_config : TrainConfig
    stage : int = -1
    next_stage_step : int = -1

    def on_collate(self, batch : list):
        tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(data['x']) for data in batch], batch_first=True, padding_value=0).to(self.train_config.device)
        x = { 'input_ids': tokens, 'attention_mask': (tokens != 0).float() } # bert forward 参数
        y = torch.tensor([data['y'] for data in batch], device=self.train_config.device)
        return x, y
    
    def on_start(self):
        self._set_stage(0)
    
    def on_step_end(self, step: int, t_loss: float):
        if step == self.next_stage_step:
             self._set_stage(self.stage + 1)

    def _set_stage(self, stage: int):
        self.stage = stage
        if stage + 1 < len(self.train_config.stage_start_step):
            self.next_stage_step = self.train_config.stage_start_step[stage + 1]
        else:
            self.next_stage_step = None

        # Setup phase learning rate
        for param_group in self.train_config.optimizer.param_groups:
            param_group['lr'] = self.train_config.stage_lr[stage]

        print(f"\n====== Stage:{stage} lr={self.train_config.stage_lr[stage]} =======")