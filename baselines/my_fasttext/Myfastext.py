import torch
import pandas as pd
import torch.utils
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import math

class Tokenizer:
    pad_token = '[PAD]'
    pad_token_id = 0

    def __init__(self, vocab_dict : dict = None, word_ngrams : int = 2):
        self.vocab_dict = vocab_dict
        self.word_ngrams = word_ngrams

    # 读取数据，创建tokenizer（建立词汇表等信息）
    @classmethod
    def create_from_data(cls, 
            data_path : str,  # 监督训练语料，格式跟原版fasttext一致
            wordNgrams : int = 2):
        data = pd.read_csv(data_path, sep='\t', header=None)
        label_id = 0
        vocab_id = 1
        label_dict = {}
        vocab_dict = { cls().pad_token : cls().pad_token_id }
        formated_data = []

        print('Loading data, creating vocab...')
        for i in tqdm(range(len(data))):
            label_name = data[0][i]
            sentences = data[1][i].split(' ')

            if label_name not in label_dict:
                label_dict[label_name] = label_id
                label_id += 1
            y = label_dict[label_name]

            x = []
            for word in cls.get_text_ngrams(sentences, wordNgrams):
                if word not in vocab_dict:
                    vocab_dict[word] = vocab_id
                    vocab_id += 1
                x.append(vocab_dict[word])

            formated_data.append((x, y))

        return cls(vocab_dict, wordNgrams), formated_data, label_dict

    # 读取数据，根据已有的词汇表进行tokenize
    def tokenize(self, data_path : str, label_dict : dict = None):
        data = pd.read_csv(data_path, sep='\t', header=None)
        formated_data = []

        for i in range(len(data)):
            label_name = data[0][i]
            sentences = data[1][i].split(' ')
            y = label_dict[label_name]
            x = [ self.vocab_dict[word] for word in self.get_text_ngrams(sentences, self.word_ngrams) if word in self.vocab_dict ]
            formated_data.append((x, y))

        return formated_data
    
    # 对文本进行N-gram切分
    @classmethod
    def get_text_ngrams(cls, words : list, wordNgrams : int):
        ngram_features = []
        # 这里是N-gram的关键，没有cls和eos的话，效果会差很多
        # N-gram是为了保留词组的局部顺序信息
        # 这里的cls和eos是为了让模型知道某个词是在开头还是结尾还是中间
        words = ['[cls]'] + words + ['[eos]'] 
        for i in range(len(words)-1):
            ngram_features.append(''.join(words[i:i+wordNgrams]))
        return ngram_features

# fastext模型
class Model(torch.nn.Module):
    def __init__(self, 
        vocab_size : int, 
        embedding_dim : int, 
        output_dim : int,
        tokenizer : Tokenizer = None,
        label_dict : dict = None,
        label_prefix : str = '__label__'
    ):
        super(Model, self).__init__()
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.label_prefix = label_prefix
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.classification = torch.nn.Linear(embedding_dim, output_dim)
            
    def forward(self, x : torch.Tensor):
        padding_mask = (x != self.tokenizer.pad_token_id).float()
        x = self.embedding(x)
        x = x * padding_mask[:, :, None]
        x = torch.mean(x, dim=1)
        logits = self.classification(x)
        return logits

    def test(self, 
        data_path : str, 
        batch_size : int = 2048*4,  
        device = 'cuda'
    ):
        return Myfastext.test(
            model=self, 
            data_path=data_path, 
            batch_size=batch_size,
            device=device)

class Myfastext:

    # fasttext监督训练
    @classmethod
    def train_supervised(cls, 
            data_path : str, 
            wordNgrams=2, 
            label : int = '__label__',
            embedding_dim : int = 100,
            epoches : int = 2,
            batch_size : int = 2048,
            lr : float = 0.2,
            device = 'cuda'
    ) -> Model:
        
        # 读取数据，创建tokenizer（建立词汇表等信息）
        tokenizer, data, label_dict = Tokenizer.create_from_data(data_path, wordNgrams)
        vocab_size = len(tokenizer.vocab_dict)
        embedding_dim = embedding_dim
        output_dim = len(label_dict)
        print(f'vocab_size={vocab_size}\nembedding_dim={embedding_dim}\noutput_dim={output_dim}')

        # 根据词汇表初始化模型
        model = Model(vocab_size, embedding_dim, output_dim, tokenizer, label_dict, label)
        model.to(device)
        model.train()
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epoches):
            dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=lambda b: cls.collate_fn(b, tokenizer.pad_token_id, device))
            total_loss = 0
            
            for x, y in tqdm(dataloader):
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

            # 学习率线性衰减
            for p in optimizer.param_groups:
                p['lr'] = lr * (1 - epoch / epoches)

            print(f'epoch={epoch}, loss={total_loss / len(dataloader)}')
        return model
    
    @classmethod
    def test(cls, 
        model : Model, 
        data_path : str, 
        batch_size : int,  
        device : str
    ):
        data = model.tokenizer.tokenize(data_path, model.label_dict)
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=lambda b: cls.collate_fn(b, model.tokenizer.pad_token_id, device))
        all_pred = []
        all_label = []
        model.eval()
        with torch.no_grad():
            for x, y in dataloader:
                y_pred = model(x)
                all_pred.extend(y_pred.argmax(dim=1).cpu().numpy())
                all_label.extend(y.cpu().numpy())
            
        return { 'samples' : len(data), 'accuracy' : accuracy_score(all_label, all_pred) }
    
    # Dataloader对每个batch的预处理函数
    # 每个batch按最长序列作为padding长度
    # 不同batch是动态padding长度的
    @classmethod
    def collate_fn(cls, batch, padding_value, device):
        x = torch.nn.utils.rnn.pad_sequence([torch.tensor(item[0]) for item in batch], batch_first=True, padding_value=padding_value)
        x = x.to(device)
        y = torch.tensor([item[1] for item in batch], device=device)
        return x, y
    