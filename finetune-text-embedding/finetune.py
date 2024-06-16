from sentence_transformers import SentenceTransformer, models, InputExample, losses, SentencesDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import random


path = ''
model_path = ''
plm_path = ''
def setup_seed(an_int):
    torch.manual_seed(an_int)
    torch.cuda.manual_seed_all(an_int)
    np.random.seed(an_int)
    random.seed(an_int)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_seed(1)

def make_input_examples_from_csv(path):
    df = pd.read_csv(path,encoding='utf-8')
    examples = []
    for _, row in df.iterrows():
        examples.append(InputExample(texts=[row['text'], row['text_pos'], row['text_neg']]))
    return examples

    
word_embedding_model = models.Transformer(plm_path, max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda:0')

train_loss = losses.MultipleNegativesRankingLoss(model=model)
input_example = make_input_examples_from_csv(path)
train_dataloader = DataLoader(input_example, shuffle=True, batch_size=32)
# # 预训练
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=10,
          warmup_steps=100,
          output_path=model_path)


