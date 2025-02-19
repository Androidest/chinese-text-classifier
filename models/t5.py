import torch
from transformers import T5Model, T5Config, BertTokenizer, T5ForSequenceClassification
from utils import TrainSchedulerBase, TrainConfigBase

# bert模型微调配置和外参
class TrainConfig(TrainConfigBase):
    random_seed : int = 1
    pretrained_path : str = 'uer/t5-base-chinese-cluecorpussmall'
    # pretrained_path : str = 'models_pretrained/T5'
    save_path : str = 'models_fine_tuned'
    model_name : str = 't5'
    start_saving_epoch : int = 4
    num_epoches : int = 7
    batch_size : int = 64 # 训练集batch_size
    eval_batch_size : int = 64 # 验证集batch_size
    test_batch_size : int = 64 # 测试集batch_size
    eval_by_steps : int = 400 # 每训练多少步进行一次验证
    dataset_cache_size : int = 100000 # 超大文本动态加载的随机缓存大小
    # 阶段性训练：不同阶段解锁不同的参数和变更学习率
    stage_start_step : list = [0, 2*2812, 4*2800] # 第几步（批次）开始切换对应的训练阶段
    stage_lr : list = [5e-5, 3e-5, 1e-5]  # 每个阶段的学习率
    def __init__(self):
        self.classes=[x.strip() for x in open(self.data_path_class).readlines()]
        self.num_classes=len(self.classes)
        self.model_tokenizer=BertTokenizer.from_pretrained(self.pretrained_path)
        self.model_config=T5Config.from_pretrained(self.pretrained_path)

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

# bert模型
class Model(torch.nn.Module):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        self.hd_size = train_config.model_config.hidden_size
        self.train_config = train_config
        self.t5 = T5Model.from_pretrained(train_config.pretrained_path)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hd_size, self.hd_size),
            torch.nn.BatchNorm1d(self.hd_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hd_size, train_config.num_classes),
        )

    def forward(self, x):
        input_ids = x['input_ids']
        eos_mask = input_ids == self.train_config.model_config.eos_token_id

        outputs = self.t5(**x)
        last_hidden_state = outputs[0]
        eos_hidden_vectors = last_hidden_state[eos_mask]
        # 取每个句子最后一个eos对应的向量(如果一个句子有多个eos隐藏向量)
        # eos_hidden_vectors = eos_hidden_vectors.view(-1, self.hd_size)[:, -1, :] 
        logits = self.classifier(eos_hidden_vectors)
        return logits
    
    def freeze_embeddings(self):
        for p in self.t5.shared.parameters():
            p.requires_grad = False

class TrainScheduler(TrainSchedulerBase):
    model : Model
    train_config : TrainConfig
    stage : int = -1
    next_stage_step : int = -1

    # 原语料数据预处理，输出结构直接用于模型训练，x会直接被传进模型的forward函数
    def on_collate(self, batch : list):
        pad_token_id = self.train_config.model_config.pad_token_id
        decoder_start_token_id = self.train_config.model_config.decoder_start_token_id
        eos_token_id = self.train_config.model_config.eos_token_id
        tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor( [decoder_start_token_id] + data['x'] + [eos_token_id]) for data in batch], 
            batch_first=True, 
            padding_value=pad_token_id).to(self.train_config.device)
        x = { 
            'input_ids': tokens,
            'attention_mask': (tokens != pad_token_id).float(),
            'decoder_input_ids': tokens
        } # bert forward 参数
        y = torch.tensor([data['y'] for data in batch], device=self.train_config.device)
        return x, y
    
    # 训练开始时
    def on_start(self):
        self._set_stage(0)
    
    # 训练完一个batch后
    def on_step_end(self, step: int, t_loss: float):
        if step == self.next_stage_step:
             self._set_stage(self.stage + 1)

    # 切换阶段
    def _set_stage(self, stage: int):
        self.stage = stage
        if stage + 1 < len(self.train_config.stage_start_step):
            self.next_stage_step = self.train_config.stage_start_step[stage + 1]
        else:
            self.next_stage_step = None
        
        # 根据阶段解冻或解冻部分参数
        if stage == 0:
            self.model.freeze_embeddings()
        
        # 设置阶段学习率
        for param_group in self.train_config.optimizer.param_groups:
            param_group['lr'] = self.train_config.stage_lr[stage]

        print(f"====== Stage:{stage} lr={self.train_config.stage_lr[stage]} =======")