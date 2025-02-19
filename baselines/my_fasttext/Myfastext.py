import torch
import pandas as pd
import torch.utils
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class Tokenizer:
    pad_token = '[PAD]'
    pad_token_id = 0

    def __init__(self, vocab_dict : dict = None, word_ngrams : int = 2):
        self.vocab_dict = vocab_dict
        self.word_ngrams = word_ngrams

    # Read data, create a tokenizer (create a vocabulary, etc.)
    @classmethod
    def create_from_data(cls, 
            data_path : str,  # Path to the training corpus in the same format that the original FastText uses.
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

    # Read data and tokenize from existing vocabularies
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
    
    # do N-gram to text
    @classmethod
    def get_text_ngrams(cls, words : list, wordNgrams : int):
        ngram_features = []
        # Here is the key to N-gram. Without cls and eos, the effect will be much worse.
        # N-gram is to preserve the local order information of the phrase
        # The cls and eos here are to let the model know whether a word is at the beginning, end or in the middle
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

    # fasttext supervised training
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
        
        tokenizer, data, label_dict = Tokenizer.create_from_data(data_path, wordNgrams)
        vocab_size = len(tokenizer.vocab_dict)
        embedding_dim = embedding_dim
        output_dim = len(label_dict)
        print(f'vocab_size={vocab_size}\nembedding_dim={embedding_dim}\noutput_dim={output_dim}')

        # Initialize the model according to the vocabulary size and output dimension
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

            # Learning rate using linear decay
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
    
    # a callback function for the Dataloader
    # Different batches uses different padding length(according to the longest sequence in the batch)
    @classmethod
    def collate_fn(cls, batch, padding_value, device):
        x = torch.nn.utils.rnn.pad_sequence([torch.tensor(item[0]) for item in batch], batch_first=True, padding_value=padding_value)
        x = x.to(device)
        y = torch.tensor([item[1] for item in batch], device=device)
        return x, y
    