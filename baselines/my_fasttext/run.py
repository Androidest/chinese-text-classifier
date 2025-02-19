print('=============== my fasttext baseline =====================')
from Myfastext import Myfastext
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
train_path = '../../data/train.txt'
dev_path =  '../../data/dev.txt'
test_path =  '../../data/test.txt'
train_preprocessed_path =  'data/train_preprocessed.txt'
dev_preprocessed_path =  'data/dev_preprocessed.txt'
test_preprocessed_path =  'data/test_preprocessed.txt'
class_path =  '../../data/class.txt'
label_prefix = '__label__' # fasttext标签必须要有个前缀，__label__是默认的

class_name = []
with open(class_path, 'r') as f:
    for name in f.readlines():
        class_name.append(name.strip())

def preprocess_data(path, new_path):
    if not os.path.exists(os.path.dirname(new_path)):
        os.mkdir(os.path.dirname(new_path))

    data = pd.read_csv(path, sep='\t', header=None)
    # 把预处理的数据写入到新文件（格式化成fasttext符合的要求）
    with open(new_path, 'w', encoding='utf8') as f:
        for i in range(len(data)):
            text = data[0][i]
            label_id = data[1][i]
            # 格式化处理
            formated_label = f'{label_prefix}{class_name[label_id]}'
            formated_text = ' '.join(text)
            # 写入行如： '__label__体育	皇 马 输 球 替 补 席 闹 丑 闻 ...'
            line = f'{formated_label}\t{formated_text}\n'
            f.write(line)

preprocess_data(train_path, train_preprocessed_path)
preprocess_data(dev_path, dev_preprocessed_path)
preprocess_data(test_path, test_preprocessed_path)

device = 'cuda'
model = Myfastext.train_supervised(
    train_preprocessed_path, # 训练集
    label=label_prefix, # 标签前缀
    wordNgrams=2,
    device=device,
    )

# 保存模型
print('dev集：', model.test(dev_preprocessed_path, device=device))
print('test集：', model.test(test_preprocessed_path, device=device))