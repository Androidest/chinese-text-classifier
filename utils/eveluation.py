from .base_classes import *
from .common import load_model
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
from tqdm import tqdm
import torch
import os

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
        dataloader = DataLoader(ds, batch_size=b_size, collate_fn=lambda b:model.collate_fn(b))

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