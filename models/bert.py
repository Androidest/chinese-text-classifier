import torch
from transformers import BertModel, BertConfig, BertTokenizer
from utils import TrainSchedulerBase, TrainConfigBase

# bert模型微调配置和外参
class TrainConfig(TrainConfigBase):
    random_seed : int = 1
    pretrained_path : str = 'ckiplab/albert-base-chinese' # 预训练模型路径, 或者Huggingface上的模型名称
    # pretrained_path : str = 'models_pretrained/albert-base-chinese'
    save_path : str = 'models_fine_tuned'
    model_name : str = 'bert'
    num_epoches : int = 3
    start_saving_epoch : int = 2 # 从第几个epoch开始保存模型，从1开始计数
    batch_size : int = 128 # 训练集batch_size
    eval_batch_size : int = 32 # 验证集batch_size
    test_batch_size : int = 64 # 测试集batch_size
    eval_by_steps : int = 200 # 每训练多少步进行一次验证
    dataset_cache_size : int = 50000 # 超大文本动态加载的随机缓存大小
    learning_rate : float = 5e-5

    def __init__(self):
        self.classes=[x.strip() for x in open(self.data_path_class).readlines()]
        self.num_classes=len(self.classes)
        self.model_tokenizer=BertTokenizer.from_pretrained(self.pretrained_path)
        self.model_config=BertConfig.from_pretrained(self.pretrained_path)

    def create_optimizer(self, model: torch.nn.Module):
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
        self.optimizer=torch.optim.AdamW(optimizer_grouped_parameters,lr=self.learning_rate)
        return self.optimizer

# bert模型
class Model(torch.nn.Module):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        self.train_config = train_config
        self.bert = BertModel.from_pretrained(train_config.pretrained_path)
        self.fc = torch.nn.Linear(train_config.model_config.hidden_size, train_config.num_classes)

    def forward(self, x):
        x = self.bert(**x)
        x = x.last_hidden_state[:, 0, :]
        return self.fc(x)

class TrainScheduler(TrainSchedulerBase):
    model : Model
    train_config : TrainConfig
    
    # 原语料数据预处理，输出结构直接用于模型训练，x会直接被传进模型的forward函数
    def on_collate(self, batch : list):
        tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([self.train_config.model_tokenizer.cls_token_id] + data['x']) for data in batch], 
            batch_first=True, 
            padding_value=0).to(self.train_config.device)
        x = { 'input_ids': tokens, 'attention_mask': (tokens != 0).float() } # bert forward 参数
        y = torch.tensor([data['y'] for data in batch], device=self.train_config.device)
        return x, y