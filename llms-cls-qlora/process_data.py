from datasets import load_dataset

import json

def load_data(name):
    if name =='cmcc':
        data = load_dataset('json',data_files={'train':'./cls_data/train.json','test':'./cls_data/test.json','val':'./cls_data/val.json'})
    if name == 'iflytek':
        data = load_dataset('csv',data_files={'train':'/data/dataset/zh/iflytek_cls/train_cls.csv','test':'/data/dataset/zh/iflytek_cls/validation_cls.csv','val':'/data/dataset/zh/iflytek_cls/validation_cls.csv'})
    if name == 'banking77':
        data = load_dataset('csv',data_files={'train':'/data/dataset/en/banking77_format/train_cls.csv','test':'/data/dataset/en/banking77_format/test_cls.csv','val':'/data/dataset/en/banking77_format/dev_cls.csv'})
    return data
def load_data_banking77():
    data = load_dataset('csv', data_files={'train':'/data/dataset/en/banking77_format/train.csv','test':'/data/dataset/en/banking77_format/test.csv','val':'/data/dataset/en/banking77_format/dev.csv'})
    return data
def load_data_csv():
    data = load_dataset('csv', data_files={'train':'../IFlyTek/train.csv','test':'../IFlyTek/test.csv','val':'../IFlyTek/validation.csv'})
    return data
def average_instruction_length(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    total_length = 0
    count = 0
    for item in data:
        if 'instruction' in item:
            length = len(item['instruction'])
            total_length += length
            count += 1
                
    return total_length / count if count > 0 else 0

def max_instruction_length(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    max_length = 0
    for item in data:
        if 'instruction' in item:
            length = len(item['instruction'])
            if length > max_length:
                max_length = length
                print(item['instruction'])
    print(max_length)
    return max_length
if __name__=='__main__':
    load_data()
    # max_instruction_length('./data/train.json')
    # print(average_instruction_length('./data/train.json'))