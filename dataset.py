import random
from torch.utils.data import IterableDataset

class CNTextClassDataset(IterableDataset):
    def __init__(self, 
                path: str, 
                config, 
                use_random: bool = True):
        self.path = path
        self.cache_size = config.dataset_cache_size
        self.tokenizer = config.model_tokenizer
        self.use_random = use_random
        self.persisted_data = [] if config.persist_data else None
        with open(self.path, 'r', encoding='utf-8') as f:
            self.line_count = sum(1 for line in f)

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
        text, label = line.split('\t')
        tokens = self.tokenizer.tokenize(text)
        tokens_emb = self.tokenizer.convert_tokens_to_ids(tokens)

        data = {
            'x': tokens_emb,
            'y': int(label),
        }

        if self.persisted_data is not None:
            self.persisted_data.append(data)

        return data