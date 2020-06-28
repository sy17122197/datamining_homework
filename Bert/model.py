import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
from data_loader import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)  # /bert_pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 每个参数都要求梯度
        self.num_class = num_class
        #self.dense = nn.Linear(768 * 2, 768)
        self.fc = nn.Linear(768, self.num_class)   # 768 -> num_class

    def forward(self, input):
        context = input[0]  # 输入的句子   (ids, seq_len, mask)
        types = input[1]
        mask = input[2]  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoded_layers, pooled = self.bert(context, token_type_ids=types, 
                              attention_mask=mask, 
                              output_all_encoded_layers=True) # 控制是否输出所有encoder层的结果
        #sequence_output = encoded_layers[6]    #encoder 为第12层的输出
        #avg_pooled = sequence_output.mean(1)             #将每个词加和取平均作为句子的表示
        #max_pooled = torch.max(sequence_output, dim=1)   #取句子中的最大词作为句子的表示
        #pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)  #将平均和最大值共同作为句子的表示
        #pooled = self.dense(pooled)   #将拼接后的表示映射回之前的长度
        logits = self.fc(pooled)   
        return logits
