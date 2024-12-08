import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=1, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)


    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        return logits
