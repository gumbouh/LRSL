# -*- coding: utf-8 -*-
"""
数据加载工具类
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import BertTokenizer
import re
import os
import pickle
import pandas as pd
import tqdm
import math
import json
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    input_ids, attn_mask, labels, sentences = zip(*batch)

    # Pad the input sequences
    input_ids = pad_sequence([torch.tensor(x).clone().detach() for x in input_ids], batch_first=True)
    attn_mask = pad_sequence([torch.tensor(x).clone().detach() for x in attn_mask], batch_first=True)
    labels = torch.tensor(labels).long()

    return input_ids, attn_mask, labels, sentences

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset[0]
        self.attn_mask = dataset[1]
        self.labels = dataset[2]
        self.processed_data = dataset[3]


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]).long(), \
               torch.tensor(self.attn_mask[index]).long(), \
               torch.tensor(self.labels[index]).long(), \
               self.processed_data[index]



class MyBatchSampler(Sampler):
    """
    按照轮次数量，从多到少顺序选取，数据要先排序
    """
    def __init__(self, batch_size, turns, drop_last=False):
        self.batch_size = batch_size
        self.turns = turns  # 每条数据有多少轮，多少句对话
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        curren_turns = self.turns[0]
        for idx in range(len(self.turns)):
            if self.turns[idx] == curren_turns and len(batch) < self.batch_size:
                batch.append(idx)
            else:
                curren_turns = self.turns[idx]
                yield batch
                batch = [idx]
            # if len(batch) == self.batch_size:
            #     yield batch
            #     batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.turns) // self.batch_size
        else:
            return (len(self.turns) + self.batch_size - 1) // self.batch_size


class Utils(object):
    def __init__(self, bert_path, max_seq_len, max_turns, batch_size, data_folder, role):
        self.max_seq_len = max_seq_len
        self.max_turns = max_turns
        self.batch_size = batch_size
        self.folder = data_folder

        # self.role = role  # complaint consult handel
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        # 疑问词
    def process(self, dialog):
        result = []
        for sentence in dialog.strip().split('[SEP]'):

            sentence = re.sub(r'[一二三四五六七八九十]+月[一二三四五六七八九十]+号', '一月一号', sentence)
            sentence = re.sub(r'[一二两三四五六七八九十]+元', 'NUM元', sentence)
            # sentence = re.sub(r'[一二两三四五六七八九十]+块[一二两三四五六七八九]*毛?[一二三四五六七八九]*', '费用', sentence)
            # sentence = re.sub(r'[一二两三四五六七八九十]个g', '流量', sentence)
            # sentence = re.sub(r'[零幺一二两三四五六七八九十百千]{2,}', 'NUM', sentence)

            if len(sentence) > 0:
                result.append(sentence)
            else:
                result.append('空')

        return result


    def read_data(self, data_type, sort=True):
        """
        读取数据
        :return: token_data, label
        """
        path = self.folder + '/{}.csv'.format(data_type)  # train/dev/test
        print('Loading {}...'.format(path))
        df = pd.read_csv(path, encoding='utf-8')
        data = []
        turns = []
        labels = []
        # lengths = []

        for dialog, label in zip(list(df['text']), list(df['label'])):
        # for dialog, label in zip(list(df['word_mf2']), list(df['c_numerical'])):
            # dialog = self.process(dialog)
            # data.append('[SEP]'.join(dialog))
            data.append(dialog)
            # turns.append(len(dialog))
            labels.append(int(label))
            
        # print(data[0],turns[0], labels[0])

        data = zip(data, labels)
        # 根据话语数排序，升序

        data = [(x[0], x[1]) for x in data]
        processed_data = [x[0] for x in data]
        # 真实句子数
        input_ids = []
        attn_mask = []
        labels = []

        for dialog, l in tqdm.tqdm(data):
            labels.append(l)
            result = self.tokenizer.encode_plus(text=dialog,
                                                text_pair=None,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                max_length=512,
                                                truncation=True,
                                                padding='max_length')

            input_ids.append(result['input_ids'])
            attn_mask.append(result['attention_mask'])
            
        return input_ids, attn_mask, labels, processed_data

    def data_loader(self, data_type, sort=True):
        input_ids, attn_mask, labels, processed_data = self.read_data(data_type, sort=sort)
        dataset = MyDataset((input_ids, attn_mask, labels, processed_data))
        if sort:
            loader = DataLoader(dataset=dataset, batch_size=self.batch_size,drop_last=True,shuffle=True,
                             collate_fn=collate_fn)
        else:
            loader = DataLoader(dataset=dataset, batch_size=1, drop_last=False, shuffle=False)
        return loader


if __name__ == "__main__":
    pass
