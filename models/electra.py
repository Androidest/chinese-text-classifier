from utils import *
import torch
from transformers import ElectraModel, ElectraConfig, ElectraTokenizer

class TrainConfig(TrainConfigBase):
    random_seed : int = 1
    pretrained_path : str = 'models_pretrained/electra_base_discriminator' # pretrained model path or Huggingface model name
    model_name : str = 'electra'
    start_saving_epoch : int = 4
    num_epoches : int = 6
    batch_size : int = 64 
    eval_batch_size : int = 64
    test_batch_size : int = 1024
    eval_by_steps : int = 400
    dataset_cache_size : int = 180000 
    # Staged training: Unlock different parameters and change the learning rate at different stages
    stage_start_step : list = [0, 1400, 2800, 3*2800] # The step at which each stage starts
    stage_lr : list = [1e-3, 1e-4, 5e-5, 1e-5] # The learning rate at each stage
    unfreeze_encoders : list = [0, 2, 4, 6, 8, 9, 10, 11] # encoder to be unlocked
    def __init__(self):
        self.classes=[x.strip() for x in open(self.data_path_class).readlines()]
        self.num_classes=len(self.classes)
        self.model_tokenizer=ElectraTokenizer.from_pretrained(self.pretrained_path)
        self.model_config=ElectraConfig.from_pretrained(self.pretrained_path)

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

class Model(ModelBase):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        hd_size = train_config.model_config.hidden_size
        self.train_config = train_config
        self.electra = ElectraModel.from_pretrained(train_config.pretrained_path)
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(hd_size, hd_size),
            torch.nn.Tanh(), 
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hd_size, hd_size),
            torch.nn.BatchNorm1d(hd_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hd_size, train_config.num_classes),
        )

    def forward(self, logits):
        last_hidden_state = self.electra(**logits)[0]
        pooler_out = self.pooler(last_hidden_state[:, 0, :])
        logits = self.classifier(last_hidden_state[:, 0, :] + pooler_out)
        return logits
    
    def collate_fn(self, batch : list):
        tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([self.train_config.model_tokenizer.cls_token_id] + data['x']) for data in batch], 
            batch_first=True, 
            padding_value=0).to(self.train_config.device)
        x = { 'input_ids': tokens, 'attention_mask': (tokens != 0).float() } # bert forward 参数
        y = torch.tensor([data['y'] for data in batch], device=self.train_config.device)
        return x, y
    
    def freeze_bert(self):
        for p in self.electra.parameters():
            p.requires_grad = False

    def unfreeze_embeddings(self):
        for p in self.electra.embeddings.word_embeddings.parameters():
            p.requires_grad = True
    
    def unfreeze_encoders(self):
        for i in self.train_config.unfreeze_encoders:
            for p in self.electra.encoder.layer[i].parameters():
                p.requires_grad = True

class TrainScheduler(TrainSchedulerBase):
    model : Model
    train_config : TrainConfig
    stage : int = -1
    next_stage_step : int = -1
    
    def on_start(self, epoch_steps: int):
        self._set_stage(0)
    
    def on_step_end(self, epoch : int, step: int, t_loss: float, t_acc : float):
        if step == self.next_stage_step:
             self._set_stage(self.stage + 1)

    def _set_stage(self, stage: int):
        self.stage = stage
        if stage + 1 < len(self.train_config.stage_start_step):
            self.next_stage_step = self.train_config.stage_start_step[stage + 1]
        else:
            self.next_stage_step = None
        
        # Unlock different parameters according to the stage
        if stage == 0:
            self.model.freeze_bert()
        elif stage == 2:
            self.model.unfreeze_encoders()
        elif stage == 3:
            self.model.unfreeze_embeddings() 
        
        # Setup phase learning rate
        for param_group in self.train_config.optimizer.param_groups:
            param_group['lr'] = self.train_config.stage_lr[stage]

        print(f"====== Stage:{stage} lr={self.train_config.stage_lr[stage]} =======")