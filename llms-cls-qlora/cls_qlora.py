
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from process_data import load_data
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
import random
from transformers import AutoModelForSequenceClassification,BitsAndBytesConfig
from transformers import MistralForSequenceClassification,BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
# from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import os
from sklearn.metrics import accuracy_score
import wandb
from trl import get_kbit_device_map
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from scipy.special import softmax
import argparse
from peft import PeftModel
import bitsandbytes as bnb
import torch.nn as nn
import json

# 给所有线性层添加adapter
def find_all_linear_names(model, train_mode):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    assert train_mode in ['lora', 'qlora']
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    # logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names


class Rerank(object):
    def __init__ (self, model_path, tag_path,dataset, var=2, topk=3):
        self.embedding_model = SentenceTransformer(model_path)
        self.dataset = dataset
        self.var = var
        self.topk = topk
        with open(tag_path, 'rb') as f:
            self.label_dict = pickle.load(f)['index2tag']

        self.label_embedding = self.get_label_embedding()
    
    def get_label_embedding(self):
        label_embedding = []
        label_num = len(self.label_dict)
        for i in range(label_num):
            label = self.label_dict[i]
            label_embedding.append(label)
        label_embedding = self.embedding_model.encode(label_embedding)
        return label_embedding
    def calculate_similarities_cpu(self, text, logits):
        if self.dataset =='banking77':
            return self.calculate_similarities_cpu_banking77(text, logits)
        
        finetune_logits = np.zeros_like(logits)
        finetune_list = [0] * len(logits)
        for i in range(len(logits)):
            topk_values = np.partition(logits[i], -self.topk)[-self.topk:]
            # topk_values = logits[i][topk_indices]
            variance = np.var(topk_values)
            if variance <= self.var:
                finetune_list[i] = 1
                sentence_emb = self.embedding_model.encode(text[i])
                sentence_emb = sentence_emb / np.linalg.norm(sentence_emb)
                # sentence_emb = sentence_emb[np.newaxis, :]  # 增加一个维度

                label_embedding = self.label_embedding / np.linalg.norm(self.label_embedding, axis=1, keepdims=True)
                # 计算余弦相似度
                similarities = sentence_emb @ label_embedding.T

                similarities = np.power(similarities, 2)

                finetune_logits[i] = softmax(logits[i])
                finetune_logits[i] = finetune_logits[i] * similarities
            else:
                finetune_list[i] = 0
                finetune_logits[i] = logits[i]
                
        return finetune_logits, finetune_list
    def calculate_similarities_cpu_banking77(self, text, logits):
        finetine_logits = np.zeros_like(logits)
        finetune_list = [0] * len(logits)
        for i in range(len(logits)):
            topk_values = np.partition(logits[i], -self.topk)[-self.topk:]
            # topk_values = logits[i][topk_indices]
            variance = np.var(topk_values)
            if variance <= self.var:
                finetune_list[i] = 1
                sentence_emb = self.embedding_model.encode(text[i])
                sentence_emb = sentence_emb / np.linalg.norm(sentence_emb)
                # sentence_emb = sentence_emb[np.newaxis, :]  # 增加一个维度

                label_embedding = self.label_embedding / np.linalg.norm(self.label_embedding, axis=1, keepdims=True)
                # 计算余弦相似度
                similarities = sentence_emb @ label_embedding.T

                #把最大的补到1
                max_val = np.max(similarities)

                similarities = np.where(similarities == max_val, 1.0, similarities)
                similarities = np.power(similarities, 3)

                finetine_logits[i] = softmax(logits[i])
                finetine_logits[i] = finetine_logits[i] * similarities
            else:
                finetune_list[i] = 0
                finetine_logits[i] = logits[i]
                
        return finetine_logits, finetune_list
    def calculate_similarities(self, text, logits):
        for i in range(len(logits)):
            topk_values, topk_predict = torch.topk(logits[i], self.topk) 
            variance = torch.var(topk_values)
            if variance <= self.var:
                sentence_emb = self.embedding_model.encode(text[i])
                sentence_emb = torch.tensor(sentence_emb, dtype=torch.float).cuda()
                sentence_emb = sentence_emb / sentence_emb.norm(dim=0, keepdim=True)
                sentence_emb = sentence_emb.unsqueeze(0)  # 增加一个维度
                        
                label_embedding = torch.tensor(self.label_embedding, dtype=torch.float).cuda()
                label_embedding = label_embedding / label_embedding.norm(dim=1, keepdim=True)
                # 计算余弦相似度
                similarities = torch.mm(sentence_emb, label_embedding.t())

                similarities = torch.pow(similarities, 2)
                
                logits[i] = F.softmax(logits[i], dim=1)
                logits[i] = logits[i] * similarities 
        return logits


# customer trainer
class CustomerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train(args, rerank):

    # data processing
    # cmcc
    data = load_data(args.dataset_name)
    # data = load_data_csv()
    # data = load_data_banking77()



    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, add_prefix_space=True)
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

    col_to_delete = ['text']
    # col_to_delete = ['sentence_sep', 'word_mf2', 'label_raw']
    def llm_preprocessing_function(examples):
        return llm_tokenizer(examples['text'], truncation=True, max_length=args.max_len)

    llm_tokenized_datasets = data.map(llm_preprocessing_function, batched=True, remove_columns=col_to_delete)
    llm_tokenized_datasets.set_format("torch")
    llm_data_collator = DataCollatorWithPadding(tokenizer=llm_tokenizer)
    # load model
    # 量化模型，会损失精度
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    bnb_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,
        # load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16
    )


    llm =  AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.llm_path,
        num_labels=args.num_labels,
        # quantization_config=bnb_config_8bit,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=get_kbit_device_map()
        # device_map='auto'
        )

    llm.config.pad_token_id = llm.config.eos_token_id

    # 量化
    llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
    # 解决输入时没有梯度下降
    if hasattr(llm, "enable_input_require_grads"):
        llm.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        llm.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        

    llm_peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout, bias=args.lora_bias, 
        target_modules=find_all_linear_names(llm,"lora")
    )
    llm = get_peft_model(llm, llm_peft_config)
    llm.print_trainable_parameters()
    
    # for metrics
    # def compute_metrics(eval_pred):

    #     logits, labels, inputs_id = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    #     # print(labels)
    #     # print(logits)
    #     inputs_id = np.where(inputs_id == -100, 2, inputs_id)
    #     original_text = llm_tokenizer.batch_decode(inputs_id, skip_special_tokens=True)
    #     rerank_logits, _ = rerank.calculate_similarities_cpu(original_text, logits)
    #     # rerank_logits = rerank.calculate_similarities_cpu_banking77(original_text, logits)
    #     predictions = np.argmax(logits, axis=-1)
    #     rerank_predictions = np.argmax(rerank_logits, axis=-1)
        
    #     acc = accuracy_score(labels, predictions)
    #     rerank_acc = accuracy_score(labels, rerank_predictions)
    #     # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
    #     return {"accuracy": acc, "rerank_accuracy": rerank_acc}
    def compute_metrics(eval_pred):

        logits, labels, inputs_id = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
        # print(labels)
        # print(logits)
        inputs_id = np.where(inputs_id == -100, 2, inputs_id)
        # original_text = llm_tokenizer.batch_decode(inputs_id, skip_special_tokens=True)
        # rerank_logits, _ = rerank.calculate_similarities_cpu(original_text, logits)
        # rerank_logits = rerank.calculate_similarities_cpu_banking77(original_text, logits)
        predictions = np.argmax(logits, axis=-1)
        # rerank_predictions = np.argmax(rerank_logits, axis=-1)
        
        acc = accuracy_score(labels, predictions)
        # rerank_acc = accuracy_score(labels, rerank_predictions)
        # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
        return {"accuracy": acc}
    # 获取当前日期和时间
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    output_dir = args.save_path + '-' + now_str
    # train
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        lr_scheduler_type= "constant",
        warmup_ratio= args.warmup_ratio,
        max_grad_norm= args.max_grad_norm,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        # fp16=True,
        bf16=True,
        gradient_checkpointing=True,
        include_inputs_for_metrics=True,
    )

    llm_trainer = CustomerTrainer(
        model=llm,
        args=training_args,
        train_dataset=llm_tokenized_datasets['train'],
        eval_dataset=llm_tokenized_datasets["val"],
        data_collator=llm_data_collator,
        compute_metrics=compute_metrics
    )

    wandb.init(config=training_args,
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                dir=args.wandb_dir,
                job_type="training",
                reinit=True)
    llm_trainer.train()
    llm_trainer.save_model(output_dir)
    # test
    test_dataset = llm_tokenized_datasets['test']   
    output = llm_trainer.predict(test_dataset)
    print(output)

def get_var(args, rerank):
    data = load_data(args.dataset_name)
    
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, add_prefix_space=True)
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    col_to_delete = ['text']
    def llm_preprocessing_function(examples):
        return llm_tokenizer(examples['text'], truncation=True, max_length=args.max_len)

    llm_tokenized_datasets = data.map(llm_preprocessing_function, batched=True, remove_columns=col_to_delete)
    llm_tokenized_datasets.set_format("torch")
    llm_data_collator = DataCollatorWithPadding(tokenizer=llm_tokenizer)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    llm = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.llm_path,
        num_labels=args.num_labels,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=get_kbit_device_map())
    
    llm.config.pad_token_id = llm.config.eos_token_id
    llm = PeftModel.from_pretrained(llm, args.model_lora_checkpoint)
    # def topk(arr, k):
    #     # 获取每行最大的k个数的索引
    #     indices = np.argpartition(arr, -k, axis=-1)[:, -k:]
    #     # 使用索引获取对应的值，并按照最后一个维度排序
    #     topk_values = np.take_along_axis(arr, np.sort(indices, axis=-1), axis=-1)
    #     return topk_values
    def get_var(labels, logits, topk, args):
        logits = torch.from_numpy(logits)
        # print(logits.size())
        values, indices = torch.topk(logits, topk, dim=-1)
        values = values.numpy()
        variances = np.var(values,axis=-1)
        # print(values.size())
        # variances = torch.var(values, dim=-1)
        # variances = variances.numpy().flatten()
        predictions = np.argmax(logits, axis=-1)
        # 找出pred与labels不一样的样本id，去计算他们的平均方差
        # 找出预测错误的样本
        wrong_var = []
        for id, (label, pred) in enumerate(zip(labels, predictions)):
            if label != pred:
                wrong_var.append(variances[id])
        avg_variance = np.mean(wrong_var)
        # incorrect_samples = predictions != labels
        # 计算预测错误的样本的平均方差
        # avg_variance = np.mean(variances[incorrect_samples])
        
        # values = values.numpy()
        topk_values = [str(v) for v in values]
        topk_list = []
        
        with open(args.tag_dict_path, 'rb') as f:
            label_dict = pickle.load(f)['index2tag']
        _, predict_3 = torch.topk(logits, 3)  # 获取前三个预测类别
        
        for predict in predict_3:
            label_topk = []
            for k in predict:
                label = label_dict[k.item()]
                label_topk.append(label)
            label_topk = ','.join(label_topk)
            topk_list.append(label_topk)
        
        true_label = list(map(lambda x: label_dict[x], labels))
        pred_label = list(map(lambda x: label_dict[x.item()], predictions))
        return avg_variance, variances, topk_values, topk_list, true_label, pred_label
        
        # for metrics
                    # 获取当前日期和时间
    now = datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    args.predict_path = args.predict_path + '-' + now_str
    def compute_metrics(eval_pred):

        logits, labels, inputs_id = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
        inputs_id = np.where(inputs_id == -100, 2, inputs_id)
        original_text = llm_tokenizer.batch_decode(inputs_id, skip_special_tokens=True)
        # rerank_logits, _ = rerank.calculate_similarities_cpu(original_text, logits)
        predictions = np.argmax(logits, axis=-1)
        # rerank_predictions = np.argmax(rerank_logits, axis=-1)
        var, var_list, values, topk_list, true_label, pred_label = get_var(labels, logits, args.rerank_topk, args)

        df = pd.DataFrame({
            'label': labels,
            # 'rerank_predictions': rerank_predictions,
            'predictions': predictions,
            'topk_predictions': topk_list,
            'true_label': true_label,
            'pred_label': pred_label,
            'var': var_list,
            'topk-var': values
        })

        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(args.predict_path + '.csv', index=False)
        
        # print(var)
        acc = accuracy_score(labels, predictions)
        # rerank_acc = accuracy_score(labels, rerank_predictions)
        # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
        # return {"accuracy": acc, "rerank_accuracy": rerank_acc, "var":var}
        return {"accuracy": acc, "var":var}

    training_args = TrainingArguments(
        do_predict=True,
        output_dir='./saves-var',
        per_device_eval_batch_size=16,
        fp16=True,
        gradient_checkpointing=True,
        include_inputs_for_metrics=True,
        )
    
    trainer = CustomerTrainer(
        model=llm,
        args=training_args,
        train_dataset=llm_tokenized_datasets['train'],
        eval_dataset=llm_tokenized_datasets["val"],
        data_collator=llm_data_collator,
        compute_metrics=compute_metrics
        )
    test_dataset = llm_tokenized_datasets['test']

    output = trainer.predict(test_dataset)
    print(output)
    
def test(args, rerank):
    data = load_data(args.dataset_name)
    
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, add_prefix_space=True)
    llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
    col_to_delete = ['text']
    def llm_preprocessing_function(examples):
        return llm_tokenizer(examples['text'], truncation=True, max_length=args.max_len)

    llm_tokenized_datasets = data.map(llm_preprocessing_function, batched=True, remove_columns=col_to_delete)
    llm_tokenized_datasets.set_format("torch")
    llm_data_collator = DataCollatorWithPadding(tokenizer=llm_tokenizer)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    llm = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.llm_path,
        num_labels=args.num_labels,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        # device_map=get_kbit_device_map()
        device_map='auto'
        )
    
    llm.config.pad_token_id = llm.config.eos_token_id
    llm = PeftModel.from_pretrained(llm, args.model_lora_checkpoint)
    def hard_sample_accuracy(labels, logits, rera_logits, rerank_list):
        total = 0
        rerank_preds = []
        origin_preds = []
        rerank_labels = []
        hard = []
        topk_logits = []
        rerank_logits = []
        ori_logits = torch.tensor(logits, dtype=torch.float).cuda()
        topk_values, _ = torch.topk(ori_logits, 3, dim=-1)
        rer_logits = torch.tensor(rera_logits, dtype=torch.float).cuda()
        rerank_topk_values, _ = torch.topk(rer_logits, 3, dim=-1)
        for id, rerank in enumerate(rerank_list):
            if rerank == 1:
                hard.append(True)
                total += 1
                rerank_labels.append(labels[id])
                origin_preds.append(np.argmax(logits[id]))
                rerank_preds.append(np.argmax(rera_logits[id]))
            else:
                hard.append(False)
                
            label_logit = []
            for k in topk_values[id]:
                label_logit.append(str(k.item()))
            label_logit = ','.join(label_logit)
            topk_logits.append(label_logit)
            rerank_label_logit = []
            for k in rerank_topk_values[id]:
                rerank_label_logit.append(str(k.item()))
            rerank_label_logit = ','.join(rerank_label_logit)
            rerank_logits.append(rerank_label_logit)
        
        origin_acc = accuracy_score(rerank_labels, origin_preds)
        rerank_acc = accuracy_score(rerank_labels, rerank_preds)
        return total, origin_acc, rerank_acc, hard, topk_logits, rerank_logits
                
        # for metrics
    def compute_metrics(eval_pred):

        logits, labels, inputs_id = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
        inputs_id = np.where(inputs_id == -100, 2, inputs_id)
        original_text = llm_tokenizer.batch_decode(inputs_id, skip_special_tokens=True)
        rerank_logits, rerank_list = rerank.calculate_similarities_cpu(original_text, logits)
        # rerank_logits = rerank.calculate_similarities_cpu_banking77(original_text, logits)
        predictions = np.argmax(logits, axis=-1)
        rerank_predictions = np.argmax(rerank_logits, axis=-1)
        hard_num, origin_acc, finetune_acc,hard, topk_logits, rerank_logits = hard_sample_accuracy(labels, logits, rerank_logits, rerank_list)
        # 将 rerank_predictions 和 predictions 转换为 DataFrame
        df = pd.DataFrame({
            'rerank_predictions': rerank_predictions,
            'predictions': predictions,
            'hard' : hard,
            'topk_logits': topk_logits,
            'rerank_logits': rerank_logits
        })
            # 获取当前日期和时间
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        output_dir = args.predict_path + '-' + now_str
        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(output_dir + '.csv', index=False)

        acc = accuracy_score(labels, predictions)
        rerank_acc = accuracy_score(labels, rerank_predictions)
        # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores. 
        result_dict =  {"seed":args.seed,"var":args.rerank_var,"accuracy": acc, "rerank_accuracy": rerank_acc, "hard_num":hard_num, "origin_acc":origin_acc, "finetune_acc": finetune_acc}
        print(result_dict)
        with open(args.predict_path + '_result.json', 'a') as f:
            json.dump(result_dict, f)
            f.write('\n')
        return result_dict

    training_args = TrainingArguments(
        do_predict=True,
        output_dir='./banking77-saves',
        per_device_eval_batch_size=16,
        fp16=True,
        gradient_checkpointing=True,
        include_inputs_for_metrics=True,
        )
    
    trainer = CustomerTrainer(
        model=llm,
        args=training_args,
        train_dataset=llm_tokenized_datasets['train'],
        eval_dataset=llm_tokenized_datasets["val"],
        data_collator=llm_data_collator,
        compute_metrics=compute_metrics
        )
    test_dataset = llm_tokenized_datasets['test']
    # test_dataset = llm_tokenized_datasets['val']

    output = trainer.predict(test_dataset)
    print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_len', type=int, default=512, help='max len of input')
    parser.add_argument('--llm_path', type=str)
    parser.add_argument('--model_lora_checkpoint')
    parser.add_argument('--predict_path', type=str)
    parser.add_argument('--text2vec_path', type=str)
    parser.add_argument('--tag_dict_path', type=str)
    parser.add_argument('--dataset_name', type=str, default=r'cmcc')
    parser.add_argument('--save_path', type=str, default=r'./saves/llama7b-qlora-cmcc')
    parser.add_argument('--wandb_project', type=str, default=r'lora-llama7b-cmcc')
    parser.add_argument('--wandb_entity', type=str)
    parser.add_argument('--wandb_dir', type=str, default=r'./log-llama7b-cmcc')
    parser.add_argument('--wandb_name', type=str, default=r'lora-cls-cmcc')
    
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=1.0, help='warmup ratio')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='max grad norm')
    parser.add_argument('--dropout', type=float, default=0.1, help='learning rate')
    parser.add_argument('--rerank_var', type=float, default=1.0, help='rerank var')
    parser.add_argument('--rerank_topk', type=int, default=3, help='rerank topk')
    parser.add_argument('--num_labels', type=int, default=34, help='dataset labels')
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=32, help='lora alpha')
    parser.add_argument('--lora_bias', type=str, default='none', help='lora bias')
    parser.add_argument("--train_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="epoch")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    
    parser.add_argument('--do_train', action='store_true', help='do training procedure?')
    parser.add_argument('--do_test', action='store_true', help='do test procedure?')
    parser.add_argument('--do_var', action='store_true', help='do var procedure?')
    
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # args.do_train = True

    # rerank = ''

    if args.do_train:
        train(args, '')
    if args.do_test:
        rerank = Rerank(model_path=args.text2vec_path, tag_path=args.tag_dict_path, 
                    dataset=args.dataset_name, var=args.rerank_var, topk=args.rerank_topk)
        test(args, rerank)
    if args.do_var:
        get_var(args, '')

