# -*- coding: utf-8 -*-
"""

"""
import os
import time
import logging
import argparse
import platform
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from bert_model import BertCLS
import data_utils_raw
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    args.model_name = args.model_name.format(args.role)

    # 日志文件夹和模型保存文件夹若不存在则创建
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    # 日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    # 输出到文件
    file_handler = logging.FileHandler(args.log_path + '{}_'.format(args.model_name).split('.')[0] + timestamp + '.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # 输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(args)

    # 数据加载
    utils = data_utils_raw.Utils(bert_path=args.bert_path,
                             max_seq_len=args.max_seq_len,
                             max_turns=args.max_turns,
                             batch_size=args.batch_size,
                             data_folder=args.data_folder,
                             role=args.role)
    # train_loader = utils.data_loader('dev')
    train_loader = utils.data_loader('train')
    dev_loader = utils.data_loader('dev')
    test_loader = utils.data_loader('test')
    # dev_loader = train_loader
    # test_loader = utils.data_loader('test')

    model = BertCLS(bert_path=args.bert_path,
                dropout=args.dropout,
                num_class=args.num_classes).to(device)
    # model.load_state_dict(torch.load(r'./model/DCR-Net.bin'))
    # 优化器与损失函数
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-8)

    # 不对bias和LayerNorm.weight做L2正则化
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # PyTorch scheduler
    # 定义训练步数
    total_steps = len(train_loader) * args.x

    # 定义warmup步数
    # warmup_steps = total_steps
    warmup_steps = int(total_steps * 0.1)

    # 定义学习率调整器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)    
    criterion = nn.CrossEntropyLoss()
    best_score = 0
    patience = 0
    training_loss = 0
    step = 0
    for epoch in range(args.epochs):
        logger.info(10 * '*' + "training epoch: {} / {}".format(epoch+1, args.epochs) + '*' * 10)
        # train mode
        model.train()
        for batch in tqdm(train_loader):
            # batch = tuple(t.to(device) for t in batch)
            input_ids, attn_mask, labels, _ = batch
            
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attn_mask)

            # loss
            loss = criterion(logits, labels)
            training_loss += loss.item()

            loss = loss / args.accumulation_steps
            # backward
            loss.backward()

            # 梯度累加
            if (step + 1) % args.accumulation_steps == 0:
                # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # log
            if (step + 1) % args.print_step == 0:
                logger.info("loss: {}".format(training_loss / args.print_step))
                training_loss = 0
            step += 1

        # 评估
        dev_score, dev_top3_score = evaluation(model=model, dev_loader=dev_loader)
        logger.info("Validation Accuracy: {}".format(dev_score))
        logger.info("Validation Top3 Accuracy: {}".format(dev_top3_score))


        if best_score < dev_score:
            logger.info("Validation Accuracy Improve from {} to {}".format(best_score, dev_score))
            torch.save(model.state_dict(), args.model_save_path+args.model_name)
            best_score = dev_score
            logger.info('--------------------')
            test_score, test_top3_score = evaluation(model=model, dev_loader=test_loader)
            logger.info("test Accuracy: {}".format(test_score))
            logger.info("test Top3 Accuracy: {}".format(test_top3_score))
            patience = 0

        else:
            logger.info("Validation Accuracy don't improve. Best Accuracy:" + str(best_score))
            patience += 1
            if patience >= args.patience:
                logger.info("After {} epochs acc don't improve. break.".format(args.patience))
                break


def evaluation(model, dev_loader):
    """
    模型在验证集上的正确率
    :param model:
    :param dev_loader:
    :return:
    """
    # eval mode
    model.eval()
    hits, totals = 0, 0
    topk_correct = 0
    with torch.no_grad():
        for batch in tqdm(dev_loader):
            
            input_ids, attn_mask, labels, _ = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            # turns = turns.to(device)
            batch = len(input_ids)
            logits = model(input_ids, attn_mask)
            _, predict = torch.max(logits, 1)
            topk_values, topk_predict = torch.topk(logits, 3)  # 获取前三个预测类别
            for i in range(len(labels)):
                if labels[i] in topk_predict[i]:
                    topk_correct += 1
            
            totals += labels.size(0)
            hits += (predict == labels).sum().item()
            
    return hits / totals, topk_correct / totals


def get_label_embedding(embedding_model, label_dict):
    label_embedding = []
    label_num = len(label_dict)
    for i in range(label_num):
        label = label_dict[i]
        label_embedding.append(label)
    label_embedding = embedding_model.encode(label_embedding)
    return label_embedding

def get_label_text(label_dict):
    label_text = []
    label_num = len(label_dict)
    for i in range(label_num):
        label = label_dict[i]
        label_text.append(label)
    # label_embedding = embedding_model.encode(label_embedding)
    return label_text

def calculate_similarities(embedding_model, label_embedding, text, logits, var=1, topk=3, rerank_all=False):
    
    logits = logits.squeeze(0)
    topk_values, topk_predict = torch.topk(logits, topk)
    variance = topk_values.cpu().numpy().var()
    
    if variance > var:
        logits = logits.unsqueeze(0)
        return logits, False
    
    logits = logits.unsqueeze(0)

    # # 判断topk logits分布突不突出，不突出就进行修正
    sentence_emb = embedding_model.encode(text)
    sentence_emb = torch.tensor(sentence_emb, dtype=torch.float).cuda()
    sentence_emb = sentence_emb / sentence_emb.norm(dim=0, keepdim=True)
    sentence_emb = sentence_emb.unsqueeze(0)

    label_embedding = torch.tensor(label_embedding, dtype=torch.float).cuda()
    label_embedding = label_embedding / label_embedding.norm(dim=1, keepdim=True)
    # 计算余弦相似度
    similarities = torch.mm(sentence_emb, label_embedding.t())
    
    # rerank key step 
    # 不同的数据集，可以对应进行相似度的调整，修改相似度对于logits的权重影响
    similarities = torch.pow(similarities, 2)
    logits = logits * similarities

    return logits, True
    
def get_finetune_dataset(args):    
    args.model_name = args.model_name.format(args.role)
    utils = data_utils_raw.Utils(bert_path=args.bert_path,
                             max_seq_len=args.max_seq_len,
                             max_turns=args.max_turns,
                             batch_size=args.batch_size,
                             data_folder=args.data_folder,
                             role=args.role)
    test_loader = utils.data_loader('train', sort=False)
    
    with open(args.tag_path, 'rb') as f:
        label_dict = pickle.load(f)['index2tag']
    model = BertCLS(bert_path=args.bert_path,
                dropout=args.dropout,
                num_class=args.num_classes).to(device)
    model.load_state_dict(torch.load(r'./model/{}'.format(args.model_name)))
    model = model.to(device)
    model.eval()

    prediction = []
    true = []
    topk_correct = 0
    topk_list= []
    var = []
    topk_logits= []
    ap_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader):

            input_ids, attn_mask, labels, text = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attn_mask)
            _, predict = torch.max(logits, 1)
            # key step
            topk_values, topk_predict = torch.topk(logits, 3)  # get top-k label 
            variance = topk_values[0].cpu().numpy().var()
            var.append(variance)
            prediction.append(predict)
            true.append(labels)
            
            ap_list.append(text)
            label_logit = []
            for k in topk_values[0]:
                label_logit.append(str(k.item()))
            label_logit = ','.join(label_logit)
            topk_logits.append(label_logit)
            
            label_topk = []
            for k in topk_predict[0]:
                label = label_dict[k.item()]
                label_topk.append(label)
            label_topk = ','.join(label_topk)
            topk_list.append(label_topk)
            # 检查前三个预测类别中是否包含正确的类别    
            for i in range(len(labels)):
                if labels[i] in topk_predict[i]:
                    topk_correct += 1
                    

    prediction = torch.cat(prediction).cpu().numpy()
    true = torch.cat(true).cpu().numpy()
    print('train set accuracy', accuracy_score(true, prediction))
    print('Top-3 accuracy', topk_correct / len(true))  # 打印Top-3准确率
    prediction = list(map(lambda x: label_dict[x], prediction))
    true = list(map(lambda x: label_dict[x], true))
    df = pd.DataFrame()
    df['predict'] = prediction
    df['true'] = true
    df['topk_predict'] = topk_list
    df['topk_logits'] = topk_logits
    df['ap_list'] = ap_list
    df ['var'] = var
    aver_var = []
    for id, row in df.iterrows():
        if(row['true'] != row['predict']):
            aver_var.append(row['var'])
    average_var = round(np.average(aver_var), 2)
    args.result_filename = args.result_filename + '-' + str(average_var)
    df.to_csv(r'./finetune-dataset/{}.csv'.format(args.result_filename), index=False, encoding='utf-8')
    
def test(args):
    args.model_name = args.model_name.format(args.role)
    log_path = './topk-var/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        
    # 日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    # 输出到文件
    file_handler = logging.FileHandler(log_path + '{}_'.format(args.model_name).split('.')[0] + timestamp + '.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # 输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info('var for rerank: {}'.format(args.var))
    args.model_name = args.model_name.format(args.role)
    utils = data_utils_raw.Utils(bert_path=args.bert_path,
                             max_seq_len=args.max_seq_len,
                             max_turns=args.max_turns,
                             batch_size=args.batch_size,
                             data_folder=args.data_folder,
                             role=args.role)
    

    test_loader = utils.data_loader('test', sort=False)
    with open(args.tag_path, 'rb') as f:
        label_dict = pickle.load(f)['index2tag']
    model = BertCLS(bert_path=args.bert_path,
                dropout=args.dropout,
                num_class=args.num_classes).to(device)
    model.load_state_dict(torch.load(r'./model/{}'.format(args.model_name)))
    embedding_model = SentenceTransformer(args.embed_model_path)
    label_embedding = get_label_embedding(embedding_model, label_dict)
    model = model.to(device)
    model.eval()

    prediction = []
    true = []
    topk_correct = 0
    topk_list= []
    topk_rerank_list = []
    sents_id = []
    sents_weights=[]
    topk = 5
    topk_logits= []
    rerank_logits = []
    ap_list = []
    hard = []
    finetune_prediction = []
    finetune_pred = []
    origin_pred = []
    finetune_label = []
    with torch.no_grad():
        for batch in tqdm(test_loader):

            input_ids, attn_mask, labels, text = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attn_mask)
            
            _, predict = torch.max(logits, 1)
            topk_values, topk_predict = torch.topk(logits, args.topk)  # 获取前三个预测类别
            
            # print(topk_predict)
            prediction.append(predict)
            true.append(labels)
            
            
            #计算修正logits
            fintune_logits, flag = calculate_similarities(embedding_model, label_embedding, text[0], logits, var=args.var, topk=args.topk, rerank_all=args.rerank_all)
            _, fintune_predict = torch.max(fintune_logits, 1)
            rerank_topk_values, rerank_topk_predict = torch.topk(fintune_logits, args.topk)
            finetune_prediction.append(fintune_predict)
            hard.append(flag)
            if flag is True:
                finetune_pred.append(fintune_predict)
                origin_pred.append(predict)
                finetune_label.append(labels)
            # 把topk个标签id转成对应的类别
            label_logit = []
            rerank_lebel_logit = []
            for k in rerank_topk_values[0]:
                rerank_lebel_logit.append(str(k.item()))
            rerank_lebel_logit = ','.join(rerank_lebel_logit)
            rerank_logits.append(rerank_lebel_logit)
            for k in topk_values[0]:
                label_logit.append(str(k.item()))
            label_logit = ','.join(label_logit)
            topk_logits.append(label_logit)
            
            label_topk = []
            for k in topk_predict[0]:
                label = label_dict[k.item()]
                label_topk.append(label)
            label_topk = ','.join(label_topk)
            topk_list.append(label_topk)
            
            label_rerank_topk = []
            for k in rerank_topk_predict[0]:
                label = label_dict[k.item()]
                label_rerank_topk.append(label)
            label_rerank_topk = ','.join(label_rerank_topk)
            topk_rerank_list.append(label_rerank_topk)
            
            # 检查前三个预测类别中是否包含正确的类别    
            for i in range(len(labels)):
                if labels[i] in topk_predict[i]:
                    topk_correct += 1
                    

    prediction = torch.cat(prediction).cpu().numpy()
    finetune_prediction = torch.cat(finetune_prediction).cpu().numpy()
    true = torch.cat(true).cpu().numpy()
    logger.info('Test set accuracy {}'.format(accuracy_score(true, prediction)))
    print('Test set accuracy', accuracy_score(true, prediction))
    finetune_label = torch.cat(finetune_label).cpu().numpy()
    finetune_pred = torch.cat(finetune_pred).cpu().numpy()
    origin_pred = torch.cat(origin_pred).cpu().numpy()
    logger.info('Top-k accuracy {}'.format(topk_correct / len(true)))
    print('finetune accuracy ', accuracy_score(true, finetune_prediction))
    logger.info('finetune accuracy {}'.format(accuracy_score(true, finetune_prediction)))
    print('----')
    print('total :', len(finetune_label))
    logger.info('total : {}'.format(len(finetune_label)))
    print('finetune ', accuracy_score(finetune_label, finetune_pred))
    logger.info('finetune {}'.format(accuracy_score(finetune_label, finetune_pred)))
    print('origin ', accuracy_score(finetune_label, origin_pred))
    logger.info('origin {}'.format(accuracy_score(finetune_label, origin_pred)))
    prediction_label = list(map(lambda x: label_dict[x], prediction))

    df = pd.DataFrame()
    df['predict'] = prediction_label
    df['prediction'] = prediction
    df['finetune_prediction'] = finetune_prediction
    df['true'] = true
    df['hard'] = hard
    df['topk_predict'] = topk_list
    df['topk_logits'] = topk_logits
    df['rerank_logits'] = rerank_logits
    df['topk_rerank_predict'] = topk_rerank_list

    df.to_csv(r'./tweet-test/{}.csv'.format(args.result_filename), index=False, encoding='utf-8')
    
    fw = open('./tweet-test/var-log.txt', 'a', encoding='utf-8')
    fw.write('-seed='+str(args.seed)+'-\n')
    fw.write('var='+str(args.var)+'\n')

    log = 'total : {}'.format(len(finetune_label)) + '\n' \
        + 'hard-acc : {}'.format(accuracy_score(finetune_label, finetune_pred)) + '\n' \
        + 'hard-origin-acc : {}'.format(accuracy_score(finetune_label, origin_pred)) + '\n' \
        + 'all-finetune-acc : {}'.format(accuracy_score(true, finetune_prediction)) + '\n' \
        + 'all-origin-acc : {}'.format(accuracy_score(true, prediction)) + '\n' 
    fw.write(log)
    
def get_valid_var(args):    
    args.model_name = args.model_name.format(args.role)
    utils = data_utils_raw.Utils(bert_path=args.bert_path,
                             max_seq_len=args.max_seq_len,
                             max_turns=args.max_turns,
                             batch_size=args.batch_size,
                             data_folder=args.data_folder,
                             role=args.role)
    test_loader = utils.data_loader('dev', sort=False)
    
    with open(args.tag_path, 'rb') as f:
        label_dict = pickle.load(f)['index2tag']
    model = BertCLS(bert_path=args.bert_path,
                dropout=args.dropout,
                num_class=args.num_classes).to(device)
    model.load_state_dict(torch.load(r'./model/{}'.format(args.model_name)))
    model = model.to(device)
    model.eval()

    prediction = []
    true = []
    topk_correct = 0
    topk_list= []
    var = []
    topk_logits= []
    ap_list = []
    perplex = []
    with torch.no_grad():
        for batch in tqdm(test_loader):

            input_ids, attn_mask, labels, text = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attn_mask)
            _, predict = torch.max(logits, 1)
            topk_values, topk_predict = torch.topk(logits, args.topk)  # 获取前三个预测类别
            variance = topk_values[0].cpu().numpy().var()
            var.append(variance)
            # print(topk_predict)
            prediction.append(predict)
            true.append(labels)
            
            ap_list.append(text)
            label_logit = []
            for k in topk_values[0]:
                label_logit.append(str(k.item()))
            label_logit = ','.join(label_logit)
            topk_logits.append(label_logit)
            
            label_topk = []
            for k in topk_predict[0]:
                label = label_dict[k.item()]
                label_topk.append(label)
            label_topk = ','.join(label_topk)
            topk_list.append(label_topk)
            # 检查前三个预测类别中是否包含正确的类别    
            for i in range(len(labels)):
                if labels[i] in topk_predict[i]:
                    topk_correct += 1
                    

    prediction = torch.cat(prediction).cpu().numpy()
    true = torch.cat(true).cpu().numpy()
    print('dev set accuracy', accuracy_score(true, prediction))
    print('Top-3 accuracy', topk_correct / len(true))  # 打印Top-3准确率
    prediction = list(map(lambda x: label_dict[x], prediction))
    true = list(map(lambda x: label_dict[x], true))
    df = pd.DataFrame()
    df['predict'] = prediction
    df['true'] = true
    df['topk_predict'] = topk_list
    df['topk_logits'] = topk_logits

    df['ap_list'] = ap_list
    df ['var'] = var
    wrong_var = []
    true_var = []
    average_var_wrong = round(np.average(wrong_var), 2)
    max_var_wrong = round(np.max(wrong_var), 2)
    min_var_wrong = round(np.min(wrong_var), 2)
    median_var_wrong = round(np.median(wrong_var), 2)  # 计算中位数

    average_var_true = round(np.average(true_var), 2)
    max_var_true = round(np.max(true_var), 2)
    min_var_true = round(np.min(true_var), 2)
    median_var_true = round(np.median(true_var), 2)  # 计算中位数

    print('aver_wrong',average_var_wrong)
    print('max_wrong',max_var_wrong)
    print('min_wrong',min_var_wrong)
    print('median_wrong',median_var_wrong)
    print('aver_true',average_var_true)
    print('max_true',max_var_true)
    print('min_true',min_var_true)
    print('median_true',median_var_true)
    
    # 计算每个区间内的数据数量
    bins_wrong = [min_var_wrong, median_var_wrong, max_var_wrong]
    bins_true = [min_var_true, median_var_true, max_var_true]

    counts_wrong, _ = np.histogram(wrong_var, bins=bins_wrong)
    counts_true, _ = np.histogram(true_var, bins=bins_true)

    print('Counts in wrong_var intervals:', counts_wrong)
    print('Counts in true_var intervals:', counts_true)
    

    args.result_filename = args.result_filename + '-'+str(args.topk)+ '-var-' + str(average_var_wrong)
    df.to_csv(r'./dev-var/{}.csv'.format(args.result_filename), index=False, encoding='utf-8')
    
def get_top_acc(args):    
    args.model_name = args.model_name.format(args.role)
    log_path = './log-test-topk-acc/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        
    # 日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    # 输出到文件
    file_handler = logging.FileHandler(log_path + '{}_'.format(args.model_name).split('.')[0] + timestamp + '.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # 输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    utils = data_utils_raw.Utils(bert_path=args.bert_path,
                             max_seq_len=args.max_seq_len,
                             max_turns=args.max_turns,
                             batch_size=args.batch_size,
                             data_folder=args.data_folder,
                             role=args.role)
    test_loader = utils.data_loader('test', sort=False)
    # test_loader = utils.data_loader('dev', sort=False)
    
    with open(args.tag_path, 'rb') as f:
        label_dict = pickle.load(f)['index2tag']
    model = BertCLS(bert_path=args.bert_path,
                dropout=args.dropout,
                num_class=args.num_classes).to(device)
    model.load_state_dict(torch.load(r'./model/{}'.format(args.model_name)))
    model = model.to(device)
    model.eval()

    prediction = []
    true = []
    top3_correct = 0
    top5_correct = 0
    top7_correct = 0
    top10_correct = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):

            input_ids, attn_mask, labels, text = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attn_mask)
            _, predict = torch.max(logits, 1)
            _, top3_predict = torch.topk(logits, 3)
            _, top5_predict = torch.topk(logits, 5)
            _, top7_predict = torch.topk(logits, 7)
            _, top10_predict = torch.topk(logits, 10)
            prediction.append(predict)
            true.append(labels)
            
            # 检查前三个预测类别中是否包含正确的类别    
            for i in range(len(labels)):
                if labels[i] in top3_predict[i]:
                    top3_correct += 1
                    
            for i in range(len(labels)):
                if labels[i] in top5_predict[i]:
                    top5_correct += 1
                    
            for i in range(len(labels)):
                if labels[i] in top7_predict[i]:
                    top7_correct += 1
            for i in range(len(labels)):
                if labels[i] in top10_predict[i]:
                    top10_correct += 1
                    

    prediction = torch.cat(prediction).cpu().numpy()
    true = torch.cat(true).cpu().numpy()
    logger.info('Top-1 accuracy {}'.format(accuracy_score(true, prediction)) )  
    logger.info('Top-3 accuracy {}'.format(top3_correct / len(true)) )  
    logger.info('Top-5 accuracy {}'.format(top5_correct / len(true)) )  
    logger.info('Top-7 accuracy {}'.format(top7_correct / len(true)) )  
    logger.info('Top-10 accuracy {}'.format(top10_correct / len(true)) )  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    bert_path = '/data/pytorch_chinese_L-12_H-768_A-12'
    parser.add_argument('--bert_path', type=str, default=bert_path, help='预训练BERT模型（Pytorch）')
    # 对输入的数据每两句拼接在一起，模型HAN不变。80*2=160
    parser.add_argument('--max_seq_len', type=int, default=100, help='句子最大长度')

    parser.add_argument('--max_turns', type=int, default=25, help='一次会话最大轮次')
    parser.add_argument('--data_folder', type=str, default=r'/data/codespace/dataset/zh/cmcc')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累加")
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--num_classes', type=int, default=34, help='类别数量')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮次')
    parser.add_argument('--patience', type=int, default=4, help='early stopping')
    parser.add_argument('--log_path', type=str, default='./log-train/', help='日志文件夹')
    parser.add_argument('--model_save_path', type=str, default='./model/', help='模型存放目录')
    parser.add_argument('--print_step', type=int, default=200, help='训练时每X步输出loss')
    parser.add_argument('--role', type=str, default='all', help='哪个类')  # complaint consult handel
    parser.add_argument('--model_name', type=str, default='HAN_adj_{}.bin', help='模型名称')
    parser.add_argument('--dropout', type=float, default=0.1, help='丢弃概率')
    parser.add_argument('--hidden_size', type=int, default=300, help='LSTM隐藏层大小')
    parser.add_argument('--do_train', action='store_true', help='do training procedure?')
    parser.add_argument('--do_get_var', action='store_true', help='do_get_var?')
    parser.add_argument('--do_topk_acc', action='store_true', help='do topk acc?')
    parser.add_argument('--do_test', action='store_true', help='do test procedure?')
    parser.add_argument('--rerank_all', action='store_true', help='rerank all samples?')
    parser.add_argument('--do_get_dataset', action='store_true', help='do test procedure?')
    parser.add_argument('--tag_path', type=str, default='./data/tag_dict.pkl', help='tag_dict存放目录')
    parser.add_argument('--embed_model_path', type=str, help='embed_model_path')
    parser.add_argument('--result_filename', type=str, required=False, default='test_result', help='测试集结果')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--var', type=float, default=1.0, help='var of rerank')
    parser.add_argument('--topk', type=int, default=3, help='topk')
    parser.add_argument('--x', type=int, default = 4, help='warmup中的epoch')

    args = parser.parse_args()
    # rerank settings
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.topk = 3

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    print('seed = ',seed)
    print(torch.cuda.is_available())
    if args.do_train:
        train(args)
    if args.do_test:
        test(args)
    if args.do_get_dataset:
        get_finetune_dataset(args)
    if args.do_get_var:
        get_valid_var(args)
    if args.do_topk_acc:
        get_top_acc(args)

