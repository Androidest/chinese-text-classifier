from utils import *
import torch
from transformers import AlbertModel, AlbertConfig, BertTokenizerFast

class TrainConfig(DistillConfigBase):
    random_seed : int = 2
    # pretrained_path : str = 'ckiplab/albert-base-chinese' # pretrained model path or Huggingface model name
    pretrained_path : str = 'models_pretrained/albert-base-chinese' # pretrained model path or Huggingface model name
    model_name : str = 'albert_dist'
    teacher_model_name : str = 'macbert'   # teacher model name for distillation
    teacher_model_acc : str = '95.22'  # to load the teacher model file with the corresponding accuracy suffix
    distilled_data_path : str = 'data_distilled/distilled_macbert.txt'
    start_saving_epoch : int = 1
    num_epoches : int = 9
    batch_size : int = 512
    eval_batch_size : int = 64
    test_batch_size : int = 1024
    eval_by_steps : int = 50
    dataset_cache_size : int = 180000
    # Staged training: Unlock different parameters and change the learning rate at different stages
    stage_start_step : list = [0, 350, 350*2, 350*4] # The step at which each stage starts
    stage_lr : list = [1e-3, 5e-4, 1e-4, 1e-5] # The learning rate at each stage

    def __init__(self):
        self.classes=[x.strip() for x in open(self.data_path_class).readlines()]
        self.num_classes=len(self.classes)
        self.model_tokenizer=BertTokenizerFast.from_pretrained(self.pretrained_path)
        self.model_config=AlbertConfig.from_pretrained(self.pretrained_path)

    def create_optimizer(self, model: torch.nn.Module):
        model.train()
        # param_optimizer=list(model.named_parameters())
        # no_decay=['bias','LayerNorm.bias','LayerNorm.weight']
        # optimizer_grouped_parameters=[
        #     {
        #         'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],
        #         'weight_decay':0.01
        #     },
        #     {
        #         'params':[p for n ,p in param_optimizer if any(nd in n for nd in no_decay) ],
        #         'weight_decay':0.0
        #     }
        # ]
        # self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        self.optimizer = torch.optim.AdamW(model.parameters())
        return self.optimizer

    def loss_fn(self, logits, labels):
        return torch.nn.CrossEntropyLoss()(logits, labels)
    
    def distill_loss_fn(self, logits, labels, teacher_logits):
        T = 2
        alpha = 0.2
        student_pred = torch.softmax(logits / T, dim=-1)
        teacher_pred = torch.softmax(teacher_logits/ T, dim=-1)

        hard_loss = torch.nn.CrossEntropyLoss()(logits, labels)
        soft_loss = torch.nn.KLDivLoss(reduction='batchmean')(student_pred, teacher_pred) * (T * T)
        return alpha * hard_loss + (1 - alpha) * soft_loss
    
class Model(ModelBase):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        config =  train_config.model_config
        hd_size = config.hidden_size
        config.num_hidden_layers = 3
        self.train_config = train_config
        self.albert = AlbertModel.from_pretrained(train_config.pretrained_path, config=config)

        self.classification = torch.nn.Sequential(
            torch.nn.Linear(hd_size, train_config.num_classes),
        )

    def forward(self, x):
        o = self.albert(**x)
        x = self.classification(o.pooler_output + o.last_hidden_state[:, 0, :])
        return x
    
    def collate_fn(self, batch : list):
        tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(data['x']) for data in batch], batch_first=True, padding_value=0).to(self.train_config.device)
        x = { 'input_ids': tokens, 'attention_mask': (tokens != 0).float() } # bert forward 参数
        y = torch.tensor([data['y'] for data in batch], device=self.train_config.device)

        if batch[0].get('logits') is not None:
            logits = torch.tensor([data['logits'] for data in batch], device=self.train_config.device)
            return x, y, logits
        
        return x, y

class TrainScheduler(TrainSchedulerBase):
    model : Model
    train_config : TrainConfig
    stage : int = -1
    next_stage_step : int = -1
    
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