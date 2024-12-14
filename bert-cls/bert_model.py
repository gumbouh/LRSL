import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel

class BertCLS(nn.Module):
    def __init__(self, bert_path, dropout, num_class):
        super(BertCLS, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, num_class)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state
        # (B*T,E,768)
        output = output[:, 0, :]  #  [CLS] specify a sentence
        output = output.squeeze(1) #
        output = self.dropout(output)
        logits = self.fc(output)
        
        return logits
