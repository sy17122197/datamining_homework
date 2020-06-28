import torch.nn as nn
import torch.nn.functional as F


class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, config.embed_dim)  #定义embedding层
        self.num_class = config.num_class
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.embed_dim, self.num_class)  # linear层

    def forward(self, x):
        x = self.word_embedding(x)   # [len(sent), batch_size, embed_dim]
        x = x.permute(1, 0, 2)       #[batch_size, len(sent), embed_dim]
        #将每句话的各个词向量取平均来作为该句子的向量表示
        pooled = F.avg_pool2d(x, (x.shape[1], 1)).squeeze(1) #[batch size, embedding_dim]
        #x = self.dropout(x)
        logits = self.fc(pooled)
        return logits
