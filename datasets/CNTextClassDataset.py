import random
from torch.utils.data import IterableDataset
import json

# for loading chinese dataset
class CNTextClassDataset(IterableDataset):
    def __init__(self, 
                path: str, 
                config, 
                use_random: bool = True,
                return_line: bool = False):
        """
        Initialize the CNTextClassDataset object.

        Args:
            path (str): The path to the dataset file.
            config: The configuration object containing dataset and model settings.
            use_random (bool, optional): Whether to use random sampling. Defaults to True.
            return_line (bool, optional): Whether to return the original line in the data. Defaults to False.
        """
        # Store the path to the dataset file
        self.path = path
        # Set the cache size for random sampling
        self.cache_size = config.dataset_cache_size
        # Initialize the tokenizer from the configuration
        self.tokenizer = config.model_tokenizer
        # Determine whether to use random sampling
        self.use_random = use_random
        # Determine whether to return the original line in the data
        self.return_line = return_line
        # Initialize the persisted data list if required
        self.persisted_data = [] if config.persist_data else None
        # Calculate the total number of lines in the dataset file
        with open(self.path, 'r', encoding='utf-8') as f:
            self.line_count = sum(1 for line in f)

    def do_not_persisted_data(self):
        self.persisted_data = None

    def __len__(self) -> int:
        return self.line_count

    def __iter__(self) -> iter:
        if self.persisted_data is not None and len(self.persisted_data) > 0:
            if self.use_random:
                random.shuffle(self.persisted_data)
            return iter(self.persisted_data)
        elif self.use_random:
            return self._random_cache_iter()
        else:
            return self._iter()
    
    def _iter(self) -> iter:
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                yield self._preprocess(line)
    
    def _random_cache_iter(self) -> iter:
        with open(self.path, 'r', encoding='utf-8') as f:
            cache = []
            for line in f:
                if len(cache) >= self.cache_size:
                    break
                data = self._preprocess(line)
                cache.append(data)

            random.shuffle(cache)

            for line in f:
                rand_index = random.randint(0, len(cache) - 1)
                yield cache[rand_index]
                data = self._preprocess(line)
                cache[rand_index] = data
            
            for data in cache:
                yield data

    def _preprocess(self, line: str) -> dict:
        line = line.split('\t')
        text = line[0]
        label = line[1].replace('\n', '')
        tokens = self.tokenizer.tokenize(text)
        tokens_emb = self.tokenizer.convert_tokens_to_ids(tokens)

        data = {
            'x': tokens_emb,
            'y': int(label),
        }

        # load teacher model's logits if exists (for model distillation)
        if len(line) > 2:
            data['logits'] = json.loads(line[2])

        # return the original line if required (for data distillation)
        if self.return_line:
            data['line'] = f'{text}\t{label}'

        if self.persisted_data is not None:
            self.persisted_data.append(data)

        return data # dict{'x', 'y', 'logits', 'line'}