import os
import torch
import random
import numpy as np
import argparse

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
arg_parser.add_argument('--acc', type=str, help='choose model accuracy', default='') # -1 means any existing accuracy
arg_parser.add_argument('--redistill', help='force re-distill data', action='store_true') # -1 means any existing accuracy

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

