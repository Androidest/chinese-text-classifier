from utils import *
import torch
from transformers import AlbertModel, AlbertConfig
from collections import Counter
import os

class Tokenizer:
    pad_token = '[PAD]'
    unk_token = '[UNK]'
    cls_token = '[CLS]'

    def get_cls_token_id(self):
        return self.vocab_dict.get(self.cls_token)

    def load(self, save_path : str):
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Vocab file not found: {save_path}")
        
        with open(save_path, 'r', encoding='utf8') as f:
            self.vocab_list = [word.strip() for word in f.readlines()]
            self.vocab_dict = { word : i for i, word in enumerate(self.vocab_list) }
            self.vocab_size = len(self.vocab_list)
            print(f"Vocab Size: {self.vocab_size}")
    
    @classmethod
    def build_vocab(cls, data_path : str, save_path : str, vocab_size : int):
        counter = Counter()
        tk = Tokenizer()
        special_tokens = [tk.pad_token, tk.unk_token, tk.cls_token]

        with open(data_path, 'r', encoding='utf8') as f:
            for line in f:
                text, _ = line.strip().split('\t')
                counter.update(text.strip())

        vocab = counter.most_common(vocab_size)
        vocab = [x[0] for x in vocab]
        vocab = special_tokens + vocab
        print(f"Vocab Size: {len(vocab)}")

        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.mkdir(folder)

        with open(save_path, 'w', encoding='utf8') as f:
            for token in vocab:
                f.write(f'{token}\n')

    def tokenize(self, text : str):
        return [token for token in text.strip()]
    
    def convert_tokens_to_ids(self, tokens : list):
        unk_token_id = self.vocab_dict.get(self.unk_token)
        return [ self.vocab_dict.get(token, unk_token_id) for token in tokens ]


class TrainConfig(DistillConfigBase):
    random_seed : int = 1
    pretrained_path : str = 'models_pretrained/albert-base-chinese' # pretrained model path or Huggingface model name
    model_name : str = 'albert_dist'
    teacher_model_name : str = 'macbert'   # teacher model name for distillation
    teacher_model_acc : str = '95.22'  # to load the teacher model file with the corresponding accuracy suffix
    distilled_data_path : str = 'data_distilled/distilled_macbert.txt'
    model_tokenizer_path : str = 'models_distilled/albert_dist/vocab.txt'
    max_vocab_size : int = 30000
    start_saving_epoch : int = 7
    num_epoches : int = 8
    batch_size : int = 64
    eval_batch_size : int = 512
    test_batch_size : int = 1024
    eval_by_steps : int = 400
    data_size = 180000
    dataset_cache_size : int = data_size
    min_lr = 1e-9
    max_lr = 1e-3
    warmup_epochs = 2

    def __init__(self):
        self.classes=[x.strip() for x in open(self.data_path_class).readlines()]
        self.num_classes=len(self.classes)
        self.model_config=AlbertConfig.from_pretrained(self.pretrained_path)

        if not os.path.exists(self.model_tokenizer_path):
            Tokenizer.build_vocab(self.data_path_train, self.model_tokenizer_path, self.max_vocab_size)

        self.model_tokenizer = Tokenizer()
        self.model_tokenizer.load(self.model_tokenizer_path)

    def create_optimizer(self, model: torch.nn.Module):
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

    def loss_fn(self, logits, labels):
        return torch.nn.CrossEntropyLoss()(logits, labels)
    
    def distill_loss_fn(self, logits, labels, teacher_logits):
        # T = 2
        # alpha = 0.5

        # student_pred = torch.log_softmax(logits / T, dim=-1)
        # teacher_pred = torch.softmax(teacher_logits/ T, dim=-1)
        # soft_loss = torch.nn.KLDivLoss(reduction='batchmean')(student_pred, teacher_pred) * (T * T)

        # hard_loss = torch.nn.CrossEntropyLoss()(logits, labels)

        # return alpha * soft_loss + (1 - alpha) * hard_loss 

        return torch.nn.MSELoss()(logits, teacher_logits)
    
class Model(ModelBase):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        config =  train_config.model_config
        config.num_hidden_layers = 3
        config.num_attention_heads = 10
        config.num_hidden_groups = 1
        config.hidden_size = 300
        config.intermediate_size = 1600
        config.hidden_dropout_prob = 0
        self.train_config = train_config
        self.albert = AlbertModel(config=config, add_pooling_layer=False)

        self.classification = torch.nn.Sequential(
            torch.nn.LayerNorm(config.hidden_size),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.hidden_size, train_config.num_classes),
        )

    def forward(self, x):
        o = self.albert(**x)
        x = o.last_hidden_state[:, 0, :]
        x = self.classification(x)
        return x
    
    def collate_fn(self, batch : list):
        prefix = [self.train_config.model_tokenizer.get_cls_token_id()]

        tokens = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(prefix + data['x']) for data in batch], 
            batch_first=True, padding_value=0).to(self.train_config.device)
        
        x = { 'input_ids': tokens, 'attention_mask': (tokens != 0).float() } # bert forward 参数
        y = torch.tensor([data['y'] for data in batch], device=self.train_config.device)

        if batch[0].get('logits') is not None:
            logits = torch.tensor([data['logits'] for data in batch], device=self.train_config.device)
            return x, y, logits
        
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