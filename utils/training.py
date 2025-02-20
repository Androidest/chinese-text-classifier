from .base_classes import *
from .common import save_model
from .eveluation import test
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics
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


