from .base_classes import *
from .common import save_model
from .eveluation import test
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

def train(
    model: torch.nn.Module,
    train_config: TrainConfigBase,
    scheduler: TrainSchedulerBase,
    ds_train: Dataset,
    ds_val: Dataset
):
    model.train()
    optimizer = train_config.create_optimizer(model)
    dataloader = DataLoader(ds_train, batch_size=train_config.batch_size, collate_fn=lambda b:model.collate_fn(b))
    e_steps = len(dataloader)
    scheduler.on_start(epoch_steps=e_steps)
    eval_by_steps_1 = train_config.eval_by_steps - 1

    def get_lr():
        return train_config.optimizer.param_groups[0]['lr']

    max_acc = 0
    acum_loss = 0
    correct_num = 0
    samples = 0
    for epoch in range(0, train_config.num_epoches):
        print(f"epoch={epoch+1}/{train_config.num_epoches} lr={get_lr():>5.2e}")
        for step, (x, y) in enumerate(tqdm(dataloader)):
            global_step = epoch * e_steps + step
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = train_config.loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            samples += len(y)
            correct_num += (y == y_pred.argmax(dim=-1)).sum().item()
            acum_loss += loss.item()
            t_loss = acum_loss / (global_step + 1)
            t_acc = correct_num / samples
            scheduler.on_step_end(epoch, global_step, t_loss, t_acc)

            # evaluate the model
            if step % train_config.eval_by_steps == eval_by_steps_1 or step == e_steps - 1:
                v_loss, v_acc = test(model, train_config, ds_val, is_eval=True)
                model.train()
                if epoch + 1 >= train_config.start_saving_epoch and v_acc > max_acc:
                    max_acc = v_acc
                    save_model(model, train_config.get_model_save_path())

                print(f"\nepoch={epoch+1}/{train_config.num_epoches} step={step+1} train_loss={t_loss:>5.2} train_acc={t_acc:>6.2%} dev_loss={v_loss:>5.2} dev_acc={v_acc:>6.2%} lr={get_lr():>5.2e}")



