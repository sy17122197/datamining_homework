from pytorch_pretrained_bert import BertModel, BertTokenizer, BertConfig, BertAdam
from torch.utils.data import *
from config import *
import torch

tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")
#训练集
input_ids_train = []       # input word ids
input_types_train = []     # segment ids
input_masks_train = []     # attention mask
label_train = []           # 标签
#验证集
input_ids_dev = []       # input word ids
input_types_dev = []     # segment ids
input_masks_dev = []     # attention mask
label_dev = []           # 标签
#测试集
input_ids_test = []       # input word ids
input_types_test = []     # segment ids
input_masks_test = []     # attention mask
label_test = []           # 标签

def read_lines(path):
    with open(path, 'r',encoding = 'utf-8') as f:
        for line in f:
            yield eval(line)
    f.close()
    

def data_set(path, train=True, dev = False, test=False):
    for items in read_lines(path):
        sent_pre = tokenizer.tokenize(items["sentence"])
        key_pre = tokenizer.tokenize(items["keywords"])     #将关键词也进行分词，有的关键词过长，为了更容易找到关联，故分词处理
        j = 0                                        #删去其中的','
        for i in range(len(key_pre)):
            if key_pre[j] == ',':
                key_pre.pop(j)
            else:
                j += 1
        tokens = []
        types = []
        
        #配置segment ids 以及将句子与关键字链接起来
        tokens.append("[CLS]")
        types.append(0)
        for token in sent_pre:
            tokens.append(token)
            types.append(0)
        tokens.append("[SEP]")
        types.append(0)
        for token in key_pre:
            tokens.append(token)
            types.append(1)
        tokens.append("[SEP]")
        types.append(1)
        #根据bert的词表转为对应下标
        ids = tokenizer.convert_tokens_to_ids(tokens)
        #配置 attention mask
        masks = [1] * len(ids)
        #当长度不足时补齐，
        if len(ids) < pad_size:
            types = types + [0] * (pad_size - len(ids))  # 填充位segment置为0
            masks = masks + [0] * (pad_size - len(ids))  # 填充位mask置0
            ids = ids + [0] * (pad_size - len(ids))
        #长度超过则切断
        else:
            types = types[:pad_size]
            masks = masks[:pad_size]
            ids = ids[:pad_size]
        assert len(ids) == len(masks) == len(types) == pad_size
        #加入至相应队列中
        if items["label"] == "115":
            items["label"] = "105"
        if items["label"] == "116":
            items["label"] = "111"
        if train:
            input_ids_train.append(ids)
            input_types_train.append(types)
            input_masks_train.append(masks)
            label_train.append([int(items["label"])-100])
        if dev:
            input_ids_dev.append(ids)
            input_types_dev.append(types)
            input_masks_dev.append(masks)
            label_dev.append([int(items["label"])-100])
        if test:
            input_ids_test.append(ids)
            input_types_test.append(types)
            input_masks_test.append(masks)
            label_test.append([int(items["label"])-100])



