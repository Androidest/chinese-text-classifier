import os
from dataset import CNTextClassDataset
from utils import find_best_model_file, save_model, load_model, arg_parser
from importlib import import_module
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    args = arg_parser.parse_args()
    print(f"========= Importing model: {args.model} ===========")
    module = import_module(f'models.{args.model}')
    TrainConfig = module.TrainConfig
    Model = module.Model
    TrainScheduler = module.TrainScheduler

    train_config = TrainConfig()
    assert os.path.exists(train_config.data_path_test)

    model = Model(train_config).to(train_config.device)
    scheduler = TrainScheduler(train_config, model)
    ds_test = CNTextClassDataset(train_config.data_path_test, train_config, use_random=False)
    max_acc, max_acc_file = find_best_model_file(train_config, model, scheduler, ds_test)

    if max_acc_file is not None:
        print(f"Found best model: max_acc={max_acc:>6.2%} max_acc_file={max_acc_file}")
        load_model(model, max_acc_file) # 加载checkpoint中的最好的模型
        save_model(model, train_config.get_model_save_path(max_acc)) # 以准确率为后缀保存模型
        print(f"Save best model: {train_config.get_model_save_path(max_acc)}")
    else:
        print(f"No checkpoint found!")