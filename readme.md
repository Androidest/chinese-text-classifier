# Introduction

---

This project focuses on text classification, featuring baseline models for quick data testing and a range of transformer models for fine-tuning.

# How to use

---

## Before Training

#### Local Pretrained Models:

1. Place your pre-trained model files into the `/models_pretrained` folder.
2. Modify the `TrainConfig.pretrained_path` in the corresponding model file under the `/models` folder. For example:

   ```python
   pretrained_path : str = 'models_pretrained/macbert_chinese_base'
   ```

#### Hugging Face Models:

If you're using a model from Hugging Face, update the `TrainConfig.pretrained_path` with the model's name. For example:

```python
pretrained_path : str = 'ckiplab/albert-base-chinese'
```

## Training the Model

To train a model, run the `train.py` script and specify the model name using the `--model=[model_name]` argument. The `model_name` should match the file name of the model located in the `/models` folder. Once training is complete, the system will automatically identify the checkpoint with the highest accuracy on the test set, and save the model as `[model_name].pth.[accuracy]`. For example, to train the `bert_opt` model:

```
python train.py --model=bert_opt
```

## Testing the Model

To test a model, use the `--model=[model_name]` argument, along with the accuracy argument `--acc=[accuracy]`. For example, to test the fine-tuned model `/models_fine_tuned/bert_opt/bert_opt.pth.95.02%`, run the following command:

```bash
python test.py --model=bert_opt --acc=95.02
```

## Finding the Best Checkpoint

To manually find the optimal model from the model's checkpoints, run following command:

```bash
python find.py --model=[model_name]
```

# 使用方法

---

## 训练前准备

#### 本地预训练模型：

1. 将你的预训练模型文件放入 `/models_pretrained` 文件夹中。
2. 如果需要网络代理（比如墙内下载Hugging Face上的模型），需要在 `utils.py` 里设置代理地址和端口，并且 `use_proxy = True` :

   ```python
   # use proxy
   use_proxy = True
   if use_proxy:
       proxy_url = '127.0.0.1:1081'
       os.environ['HTTP_PROXY'] = f'http://{proxy_url}'
       os.environ['HTTPS_PROXY'] = f'http://{proxy_url}'
       print(f"Using proxy on: {proxy_url}")
   ```
3. 修改 `/models` 文件夹中相应模型文件中的 `TrainConfig.pretrained_path` 配置。例如：

   ```python
   pretrained_path : str = 'models_pretrained/macbert_chinese_base'
   ```

#### Hugging Face 模型：

如果你使用的是 Hugging Face 上的模型，需要在 `TrainConfig.pretrained_path` 中更新为模型的名称。例如：

```python
pretrained_path : str = 'ckiplab/albert-base-chinese'
```

## 训练模型

要训练一个模型，请运行 `train.py` 脚本，并使用 `--model=[model_name]` 参数指定模型名称。`model_name` 应该与 `/models` 文件夹中的模型文件名一致。训练完成后，系统会自动识别在测试集上准确率最高的checkpoint，并将模型保存为 `[model_name].pth.[accuracy]`。例如，训练 `bert_opt` 模型的命令：

```bash
python train.py --model=bert_opt
```

## 测试模型

要测试模型，请使用 `--model=[model_name]` 参数，并附加准确率参数 `--acc=[accuracy]`。例如，要测试微调后的模型文件 `/models_fine_tuned/bert_opt/bert_opt.pth.95.02%`，请运行以下命令：

```bash
python test.py --model=bert_opt --acc=95.02
```

## 寻找最优checkpoint

如果你希望手动查找模型的最佳检查点，可以运行以下命令：

```bash
python find.py --model=[model_name]
```
