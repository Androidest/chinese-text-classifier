import torch
import json
import os

class TrainConfigBase:
    # common parameters
    random_seed : int = 1
    data_path_train : str = 'data/train.txt'
    data_path_val : str = 'data/dev.txt'
    data_path_test : str = 'data/test.txt'
    data_path_class : str = 'data/class.txt'
    pretrained_path : str = 'models_pretrained/xxx' # pretrained model path or Huggingface model name
    save_path : str = 'models_fine_tuned'
    model_name : str = 'xxx'
    num_epoches : int = 8
    start_saving_epoch : int = 1 # Save the model from the first epoch and count from 1
    batch_size : int = 128 # training batch_size
    eval_batch_size : int = 128 # evel batch_size
    test_batch_size : int = 1024 # test batch_size
    eval_by_steps : int = 200 # evaluate the model every 'eval_by_steps' steps when training
    dataset_cache_size : int = 50000 # Random cache size for large text dynamic loading
    persist_data : bool = True # whether to load all the data into memory, otherwise it will be loaded dynamically by chunks
    optimizer = None # this will be set by the model's create_optimizer function
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    def create_optimizer(self, model: torch.nn.Module):
        raise NotImplementedError

    def loss_fn(self, logits, labels):
        return torch.nn.CrossEntropyLoss()(logits, labels)
    
    # save and load functions
    def save_path_acc(self, path, acc = None):
        if acc == '-1':
            # -1 means any existing accuracy
            files = search_files_starting_with_name(os.path.dirname(path), os.path.basename(path))
            if len(files) == 0:
                raise FileNotFoundError(f"Model file not found: {path}")
            return files[0]

        if acc is None or acc == '':
            return path
        if isinstance(acc, str):
            return f"{path}.{acc}%"
        if isinstance(acc, int):
            return f"{path}.{acc}.00%"
        if isinstance(acc, float):
            if acc <= 1:
                return f"{path}.{acc:>6.2%}"
            else:
                return f"{path}.{acc:.2f}%"
        return path

    def get_model_save_path(self, acc = None):
        path = f"{self.save_path}/{self.model_name}/{self.model_name}.pth"
        return self.save_path_acc(path, acc)

    def get_config_save_path(self, acc = None):
        path = f"{self.save_path}/{self.model_name}/{self.model_name}.config"
        return self.save_path_acc(path, acc)

    def get_checkpoint_save_path(self, epoch : int, step : int):
        return f"{self.save_path}/{self.model_name}/checkpoints/{self.model_name}.pth.{epoch}_{step}"

    def save(self, path: str):
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        dic = {}
        bases = (str, float, int, bool, list, dict, tuple, set)
        keys = list(self.__annotations__.keys()) + list(self.__dict__.keys())
        for k in keys:
            v = self.__getattribute__(k)
            if (isinstance(v, bases)):
                dic[k] = v
        json_str = json.dumps(dic, indent=4)
        with open(path, 'w') as f:
            f.write(json_str)

    def load(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'r') as f:
            keys = list(self.__annotations__.keys()) + list(self.__dict__.keys())
            dic = json.loads(f.read())
            for k, v in dic.items():
                if k in keys:
                    self.__setattr__(k, v)
            return self
        

class DistillConfigBase(TrainConfigBase):
    save_path : str = 'models_distilled'
    teacher_model_name : str = 'xxx'   # teacher model name for distillation
    teacher_model_acc : str = '95.02'  # to load the teacher model file with the corresponding accuracy suffix
    distilled_data_path : str = 'data_distilled/distilled_xxx.txt'
    temperature_div : float = 1.0
    temperature : float = 1.0

    def distill_loss_fn(self, logits, labels, teacher_logits):
        raise NotImplementedError

def search_files_starting_with_name(path : str, name : str, recursive : bool = False):
    matching_files = []
    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if file.startswith(name):
                matching_files.append(os.path.join(root, file))
        if not recursive:
            break
    return matching_files