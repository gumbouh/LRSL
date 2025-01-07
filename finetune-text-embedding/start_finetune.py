import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from sentence_transformers import SentenceTransformerTrainer,SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformer, models, InputExample, losses, SentencesDataset
from torch.utils.data import DataLoader
from datasets import Dataset
import pandas as pd
import numpy as np
import torch
import random
def setup_seed(an_int):
    torch.manual_seed(an_int)
    torch.cuda.manual_seed_all(an_int)
    np.random.seed(an_int)
    random.seed(an_int)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def make_input_examples_from_csv(data_path):
    df = pd.read_csv(data_path,encoding='utf-8')
    examples = []
    for _, row in df.iterrows():
        # examples.append(InputExample(texts=[row['anchor'], row['positive'], row['negative']]))
        examples.append(InputExample(texts=[row['text'], row['text_pos'], row['text_neg']]))
    return examples

def get_model(plm_path):
    
    if "roberta" in plm_path:
        word_embedding_model = models.Transformer(plm_path, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda')
    else:
        # m3e bge
        model = SentenceTransformer(plm_path, device='cuda')
    return model
def sft_model(model, loss_fnt, dataloader, save_path, epochs=2):
    
    model.fit(train_objectives=[(dataloader, loss_fnt)],
            epochs=epochs,
            warmup_steps=100,
            output_path=save_path)
    
    
if __name__ == '__main__':
    seed = sys.argv[1]
    data_path = sys.argv[2]
    te_path = sys.argv[3]
    save_path = sys.argv[4]
    
    setup_seed(seed)
    model = get_model(te_path)

    train_loss = losses.MultipleNegativesRankingLoss(model=model,scale=1)
    input_example = make_input_examples_from_csv(data_path)
    train_dataloader = DataLoader(input_example, shuffle=True, batch_size=4)
    sft_model(model, train_loss, train_dataloader, save_path, epochs=2)
    
    # data_path = '/data/gumbou/codespace/work-backup/LRSL/non-llms/bert/finetune-dataset/bert-tweet-train-semantic-1.93-seed1.csv'
    data_path = '/data/gumbou/codespace/work-backup/LRSL/multi-label/finetune-dataset/bert-aapd-train-semantic-top5.csv'
    # data_path = '/data/gumbou/codespace/work-backup/LRSL/non-llms/bert/finetune-dataset/bert-tweet-train-1.93-seed1.csv'
    # save_path = '/data/gumbou/codespace/work-backup/text2vec/roberta-tweet-seed1-origin-label-5epoch-bge-base'
    # save_path = '/data/gumbou/codespace/work-backup/text2vec/roberta-tweet-seed1-enhance-label-5epoch-bge-base'
    save_path = '/data/gumbou/codespace/work-backup/LRSL/llms-cls-qlora/multi-label/te/bge'
    # plm_path = '/data/public_model/roberta-base'
    # plm_path = '/data/public_model/text_embedding/bge-base-en-1.5'
    plm_path = '/data/public_model/text_embedding/bge-m3'
    # plm_path = '/data/public_model/chinese_roberta_wwm_ext'






    # train_loss = losses.TripletLoss(model=model)

    # df = pd.read_csv(data_path)
    # data = Dataset.from_pandas(df)

    # print(data)

