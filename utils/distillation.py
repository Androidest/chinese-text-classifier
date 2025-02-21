from datasets import *
from sklearn import metrics
import os
from importlib import import_module
from .eveluation import *
from .base_classes import *
from .common import *
from torch.utils.data import DataLoader
from tqdm import tqdm

def distill_data(student_train_config):
    """
    Requirements:
        1. The dataset needs to return the original line from the corpus
    """
    # distilation config
    teacher_model_name = student_train_config.teacher_model_name
    teacher_model_acc = student_train_config.teacher_model_acc
    distilled_data_path = student_train_config.distilled_data_path
    # load dataset with no random shuffle and return the original line
    ds_train = CNTextClassDataset(student_train_config.data_path_train, student_train_config, use_random=False, return_line=True)
    ds_train.do_not_persisted_data()

    # create folder if not exists
    folder = os.path.dirname(distilled_data_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(distilled_data_path, 'w', encoding='utf8') as f:
        # load teacher model
        module = import_module(f'models.{teacher_model_name}')
        teacher_train_config = module.TrainConfig()
        teacher_model = module.Model(teacher_train_config).to(teacher_train_config.device)
        teacher_model = load_model(teacher_model, teacher_train_config.get_model_save_path(teacher_model_acc))
        dataloader = DataLoader(ds_train, batch_size=teacher_train_config.test_batch_size,
                               shuffle=False,
                               collate_fn=lambda b:(teacher_model.collate_fn(b), b))

        # distill the data with the teacher model by saving the output logits
        teacher_model.eval()
        with torch.no_grad():
            lines = ''
            for (x, y), b in tqdm(dataloader, desc='Distilling data'):
                logits_batch = teacher_model(x)
                logits_batch = logits_batch.detach().cpu().tolist()
                for i, data in enumerate(b):
                    logits = logits_batch[i]
                    lines += f'{data['line']}\t{logits}\n' # concatenate the original line and the logits with tab
                
                # save the data to file every 1M lines
                if len(lines) > 1000000: 
                    f.write(lines)
                    lines = ''

def distill_model(
    model: torch.nn.Module,
    train_config: TrainConfigBase,
    scheduler: TrainSchedulerBase,
    ds_train: Dataset,
    ds_val: Dataset
):
    """
    Requirements:
        1. The collate_fn of student model needs to return the logits from the teacher model
    """
    model.train()
    optimizer = train_config.create_optimizer(model)
    dataloader = DataLoader(ds_train, batch_size=train_config.batch_size, collate_fn=lambda b:model.collate_fn(b))
    scheduler.on_start()
    eval_by_steps_1 = train_config.eval_by_steps - 1
    optimizer.zero_grad()

    e_steps = len(dataloader)
    for epoch in range(1, train_config.num_epoches + 1):
        print(f"Epoch: {epoch}/{train_config.num_epoches}")
        for step, (x, y, logits) in enumerate(tqdm(dataloader)):
            y_pred = model(x)
            loss = train_config.distill_loss_fn(y_pred, y, logits)
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