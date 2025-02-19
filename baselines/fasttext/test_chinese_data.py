print('=============== fasttext baseline =====================')
import fasttext
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
label_prefix = '__label__' # default prefix label

class_name = []
with open(class_path, 'r') as f:
    for name in f.readlines():
        class_name.append(name.strip())

# preprocess chinese dataset
def preprocess_data(path, new_path):
    if not os.path.exists(os.path.dirname(new_path)):
        os.mkdir(os.path.dirname(new_path))

    data = pd.read_csv(path, sep='\t', header=None)
    # Write the preprocessed data to a new file (formatted as fasttext required)
    with open(new_path, 'w', encoding='utf8') as f:
        for i in range(len(data)):
            text = data[0][i]
            label_id = data[1][i]
            # formatting
            formated_label = f'{label_prefix}{class_name[label_id]}'
            formated_text = ' '.join(text)
            # '__label__体育	皇 马 输 球 替 补 席 闹 丑 闻 ...'
            line = f'{formated_label}\t{formated_text}\n'
            f.write(line)

preprocess_data(train_path, train_preprocessed_path)
preprocess_data(dev_path, dev_preprocessed_path)
preprocess_data(test_path, test_preprocessed_path)

model = fasttext.train_supervised(
    train_preprocessed_path, 
    autotuneValidationFile=dev_preprocessed_path, # Automatic parameter validation set
    autotuneDuration=600, # auto tuning time
    wordNgrams=2, 
    label=label_prefix, 
    verbose=3)

model.save_model('ft_model.bin')
print('dev set:', model.test(dev_preprocessed_path))
print('test set:', model.test(test_preprocessed_path))