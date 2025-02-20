from utils import *
from datasets import *
from importlib import import_module
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # parse args
    args = arg_parser.parse_args()
    print(f"========= Importing model: {args.model} ===========")

    # import models dynamically
    module = import_module(f'models.{args.model}')
    TrainConfig = module.TrainConfig
    Model = module.Model
    TrainScheduler = module.TrainScheduler

    print(f"=================== Start training =======================")
    train_config = TrainConfig()
    train_config.save(train_config.get_config_save_path())
    assert os.path.exists(train_config.data_path_train)
    assert os.path.exists(train_config.data_path_val)
    assert os.path.exists(train_config.data_path_test)
    set_seed(train_config.random_seed)
    model = Model(train_config).to(train_config.device)
    
    scheduler = TrainScheduler(train_config, model)
    ds_train = CNTextClassDataset(train_config.data_path_train, train_config)
    ds_val = CNTextClassDataset(train_config.data_path_val, train_config, use_random=False)
    train(model, train_config, scheduler, ds_train, ds_val)

    print(f"=============== Start finding best model =================")
    ds_test = CNTextClassDataset(train_config.data_path_test, train_config, use_random=False)
    max_acc, max_acc_file = eveluation(train_config, model, scheduler, ds_test, verbose=True)
    print(f"Found best model: max_acc={max_acc:>6.2%} max_acc_file={max_acc_file}")

    print(f"=================== Test best model ======================")
    model = load_model(model, max_acc_file)
    test_loss, test_acc, report, confusion = test(model, train_config, scheduler, ds_test, return_all=True, verbose=True)
    save_model(model, train_config.get_model_save_path(test_acc))
    train_config.save(train_config.get_config_save_path(test_acc))
    print("Test result:")
    print(f"test_loss={test_loss:>5.2} test_acc={test_acc:>6.2%}")
    print("Precision,recall and F1-score:")
    print(report)
    print("confusion Matrix:")
    print(confusion)
