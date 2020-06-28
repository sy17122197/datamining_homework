import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, config.embed_dim)  #定义embedding层
        self.rnn = nn.LSTM(config.embed_dim, config.hidden_size, 
                           num_layers = config.num_layers,    #RNN神经网络深度
                           bidirectional=True,                #双向循环RNN
                           dropout = config.dropout)         
        self.dropout = nn.Dropout(config.dropout)
        self.num_class = config.num_class
        self.fc = nn.Linear(config.hidden_size*2, self.num_class)  
        #以两个hidden state的值来表示句子

    def forward(self, x):
        x = self.word_embedding(x)   # [len(sent), batch_size, embed_dim]
        x = self.dropout(x)
        output, (hidden, cell) = self.rnn(x)   #放入双层的循环rnn得到hidden
        #output = [len(sen), batch_size, hidden_size * num directions]
        #hidden = [num_layers * num_directions, batch_size, hidden_size]
        #cell = [num_layers * num directions, batch_size, hidden_size]
        #取前向传播的最后一个hidden state以及反向传播的第一个hidden state,拼接起来表示句子
        # [batch_size, hidden_size * num_directions]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.dropout(hidden)
        logits = self.fc(x.squeeze(0))
        return logits
