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

    print(f"=================== Start Distilling Data =======================")
    train_config = TrainConfig()
    set_seed(train_config.random_seed)
    train_config.save(train_config.get_config_save_path())
    assert os.path.exists(train_config.data_path_train)
    assert os.path.exists(train_config.data_path_val)
    assert os.path.exists(train_config.data_path_test)
    
    # if distilled data does not exist or args.redistill=True redistill the data with the teacher model
    if not os.path.exists(train_config.distilled_data_path) or args.redistill:
        print(f"Teacher model [{train_config.teacher_model_name}], Distilled data path [{train_config.distilled_data_path}]")
        distill_data(train_config)

    model = Model(train_config).to(train_config.device)
    scheduler = TrainScheduler(train_config, model)
    # use the distilled data to train the student model (path = train_config.distilled_data_path)
    ds_train = CNTextClassDataset(train_config.distilled_data_path, train_config)
    ds_val = CNTextClassDataset(train_config.data_path_val, train_config, use_random=False)
    distill_model(model, train_config, scheduler, ds_train, ds_val)

    print(f"=============== Start finding best model =================")
    ds_test = CNTextClassDataset(train_config.data_path_test, train_config, use_random=False)
    max_acc, max_acc_file = find_best_model_file(train_config, model, scheduler, ds_test, verbose=True)
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
