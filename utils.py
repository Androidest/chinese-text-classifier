import os
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
from tqdm import tqdm
import random
import numpy as np
import json
import argparse
import os

# use proxy if needed
use_proxy = False
if use_proxy:
    proxy_url = '127.0.0.1:1081'
    os.environ['HTTP_PROXY'] = f'http://{proxy_url}'
    os.environ['HTTPS_PROXY'] = f'http://{proxy_url}'
    print(f"Using proxy on: {proxy_url}")

# common argument parser for entry scripts
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model', type=str, help='choose a model', default='bert_opt') # default is the bert_opt model
arg_parser.add_argument('--acc', type=str, help='choose model accuracy', default='-1') # -1 means any existing accuracy

class TrainConfigBase:
    # common parameters
    random_seed : int = 1
    data_path_train : str = 'data/train.txt'
    data_path_val : str = 'data/dev.txt'
    data_path_test : str = 'data/test.txt'
    data_path_class : str = 'data/class.txt'
    pretrained_path : str = 'models_pretrained/xxx'
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
    loss_fn = torch.nn.CrossEntropyLoss() 
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    def create_optimizer(self, model: torch.nn.Module):
        raise NotImplementedError
    
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

class TrainSchedulerBase:
    def __init__(self, train_config: TrainConfigBase, model: torch.nn.Module):
        self.train_config = train_config
        self.model = model
    
    # preprocess data from the dataset, 
    # the output structure is directly used for model training, 
    # and x will be directly passed into the forward function of the model
    def on_collate(self, batch : list):
        pass

    # on start training
    def on_start(self):
        pass
    
    # on end of a batch
    def on_step_end(self, step: int, t_loss: float):
        pass

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(model: torch.nn.Module, save_path: str):
    folder = os.path.dirname(save_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model.state_dict(), save_path)

def load_model(model: torch.nn.Module, save_path: str):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Model file not found: {save_path}")
    model.load_state_dict(torch.load(save_path))
    return model

def search_files_starting_with_name(path : str, name : str, recursive : bool = False):
    matching_files = []
    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            if file.startswith(name):
                matching_files.append(os.path.join(root, file))
        if not recursive:
            break
    return matching_files

def train(
    model: torch.nn.Module, 
    train_config: TrainConfigBase, 
    scheduler: TrainSchedulerBase, 
    ds_train: Dataset, 
    ds_val: Dataset
):
    model.train()
    loss_fn = train_config.loss_fn
    optimizer = train_config.create_optimizer(model)
    dataloader = DataLoader(ds_train, batch_size=train_config.batch_size, collate_fn=lambda b:scheduler.on_collate(b))
    scheduler.on_start()
    eval_by_steps_1 = train_config.eval_by_steps - 1
    optimizer.zero_grad()

    e_steps = len(dataloader)
    for epoch in range(1, train_config.num_epoches + 1):
        print(f"Epoch: {epoch}/{train_config.num_epoches}")
        for step, (x, y) in enumerate(tqdm(dataloader)):

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            t_loss = loss.item()
            scheduler.on_step_end((epoch-1) * e_steps + step, t_loss)

            if step % train_config.eval_by_steps == eval_by_steps_1:
                # evaluate the model
                t_acc = metrics.accuracy_score(y.cpu(), y_pred.argmax(dim=-1).cpu())
                v_loss, v_acc = test(model, train_config, scheduler, ds_val, is_eval=True)
                print(f"\nepoch={epoch}/{train_config.num_epoches} step={step+1} train_loss={t_loss:>5.2} train_acc={t_acc:>6.2%} dev_loss={v_loss:>5.2} dev_acc={v_acc:>6.2%}")
                
                # save checkpoint
                if epoch >= train_config.start_saving_epoch:
                    save_model(model, train_config.get_checkpoint_save_path(epoch, step))

                model.train()
    model.eval()
    save_model(model, train_config.get_checkpoint_save_path(train_config.num_epoches, e_steps))

def test(
    model: torch.nn.Module, 
    train_config: TrainConfigBase, 
    scheduler: TrainSchedulerBase, 
    ds: Dataset, 
    return_all: bool = False, 
    verbose: bool = False,
    is_eval: bool = False
):
    model.eval()
    with torch.no_grad():
        b_size = train_config.eval_batch_size if is_eval else train_config.test_batch_size
        dataloader = DataLoader(ds, batch_size=b_size, collate_fn=lambda b:scheduler.on_collate(b))

        if verbose:
            dataloader = tqdm(dataloader)

        labels_all = []
        predict_all = []
        losses = 0
        for (x, y) in dataloader:
            y_pred = model(x)
            losses += train_config.loss_fn(y_pred, y).item()
            labels_all.extend(y.cpu().tolist())
            predict_all.extend(y_pred.argmax(dim=-1).cpu().tolist())

        acc = metrics.accuracy_score(labels_all, predict_all)
        loss = losses / len(dataloader)

        if return_all:
            report = metrics.classification_report(labels_all, predict_all, target_names=train_config.classes, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return loss, acc, report, confusion

        return loss, acc

def find_best_model_file(
    train_config : TrainConfigBase, 
    model: torch.nn.Module, 
    scheduler: TrainSchedulerBase, 
    ds_test: Dataset, 
    verbose: bool =True
):
    max_acc = 0
    max_acc_file = None
    # get the model's checkpoint folder
    folder = os.path.dirname(train_config.get_checkpoint_save_path(0, 0))

    if not os.path.exists(folder):
        return max_acc, max_acc_file

    # iterate the checkpoint files
    for filename in os.listdir(folder):
        file_ext = filename.split('.')[-1]
        if '_' not in file_ext:
            continue

        file_path = f'{folder}/{filename}'
        model = load_model(model, file_path)
        _, test_acc = test(model, train_config, scheduler, ds_test, return_all=False, verbose=verbose)

        # find the checkpoint file with the highest accuracy
        if test_acc > max_acc:
            max_acc = test_acc
            max_acc_file = file_path

        print(f"filename={filename} test_acc={test_acc:>6.2%} [max_acc={max_acc:>6.2%}]")
        
    return max_acc, max_acc_file