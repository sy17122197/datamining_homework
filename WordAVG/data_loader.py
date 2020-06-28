# -*- coding: utf-8 -*-
import os
import math
import torch
import random
import json
from collections import Counter
import jieba

PAD = '<pad>'  # 0
UNK = '<unk>'  # 1
BOS = '<s>'   # 2
EOS = '</s>'  # 3
# 输入： <s> I eat sth .
# 输出： I eat sth  </s>

# encoding=utf-8
# import jieba

# strs=["我来到北京清华大学","乒乓球拍卖完了","中国科学技术大学"]
# for str in strs:
#     seg_list = jieba.cut(str,use_paddle=True) # 使用paddle模式
#     print("Paddle Mode: " + '/'.join(list(seg_list)))

# seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
# print("Full Mode: " + "/ ".join(seg_list))  # 全模式

# seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
# print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

# seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
# print(", ".join(seg_list))

# seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
# print(", ".join(seg_list))


def read_lines(path):
    """
    {"label": "102",
    "label_desc": "news_entertainment",
    "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物",
    "keywords": "江疏影,美少女,经纪人,甜甜圈"}
    """
    with open(path, 'r',encoding = 'utf-8') as f:
        for line in f:
        #yield在python里面是一个生成器，当使用一个yield的时候，对应的函数就是一个生成器；
        #生成器的功能就是在yield的区域进行迭代处理，下一次迭代时将从上一次迭代遇到的yield后面的代码下一行开始执行
        #yield是一个类似return的关键字，迭代一次遇到yield时就返回yield后面的值
        #好处：1.不会将所有数据取出来存入内存，而是返回了一个对象，可以通过对象获取数据，用多少取多少，进而节省内存空间
        #     2.除了能返回一个值，还不会终止循环的运行；
            yield eval(line)
    f.close()


class Vocab(object):
    def __init__(self, specials=[PAD, UNK, BOS, EOS], config=None,  **kwargs):
        self.specials = specials  #特殊词,UNK为出现频数低于阈值的字,PAD为填充字,BOS为句子的开始，EOS为句子的结尾
        self.counter = Counter()  #用于记录词频
        self.stoi = {}            #根据字符串找到对应的下标
        self.itos = {}            #根据下标找到对应的字符串
        self.weights = None
        self.min_freq = config.min_freq  #出现的频数下限

    def make_vocab(self, dataset):   #生成词表 + stoi + itos 为后续提供一个查找依据
        for x in dataset:
            if x != [""]:  #只不统计“ ”，其余标点符号并未排除在外
                self.counter.update(x)      #更新被统计对象的元素,相当于每读入一句将根据该词更新统计的出现情况
        if self.min_freq > 1:
            #用于过滤序列，当出现频数大于5时才做保留，否则视作UNK或略去；
            #从self.counter中选出满足 x[1] >= self.min_freq的键值对。
            self.counter = {w: i for w, i in filter(lambda x: x[1] >= self.min_freq, self.counter.items())}
        self.vocab_size = 0                   #词表的长度
        for w in self.specials:               #先将词表的前几位分配给特殊词
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1

        for w in self.counter.keys():        #将之前统计的词频情况（键值对中的key）对应词放入词表
            self.stoi[w] = self.vocab_size
            self.vocab_size += 1
        # stoi 提供了根据相应词找到其下标索引的列表

        self.itos = {i: w for w, i in self.stoi.items()}
        #itos 提供了根据下标索引找到对应字词的列表
        
    def __len__(self):   #将每个词加入词表时顺便统计了词表的长度即 self.vocab_size
        return self.vocab_size


class DataSet(list):     #获取数据集
    def __init__(self, *args, config=None, is_train=True, dataset="train"):
        self.config = config       #引入配置文件，需要其中的参数
        self.is_train = is_train   #判断是否为训练数据集
        self.dataset = dataset     #数据集标识符，指明提取的数据路径
        self.data_path = os.path.join(self.config.data_path, dataset + ".json")
        #super()函数是用于调用父类（超类）的一个方法，继承相应函数及属性
        super(DataSet, self).__init__(*args)

    def read(self):   #读取对应文件中的内容
        for items in read_lines(self.data_path):
            #以句子与标签形成对应的二元组
            sent_pre = tuple(jieba.cut(items["sentence"], cut_all=False))  #将句子利用jieba库函数按词就行划分
            key_pre = list(jieba.cut(items["keywords"]))     #将关键词也进行分词，有的关键词过长，为了更容易找到关联，故分词处理
            j = 0                                        #删去其中的','
            for i in range(len(key_pre)):
                if key_pre[j] == ',':
                    key_pre.pop(j)
                else:
                    j += 1
            key_pre = tuple(key_pre)
            sent = sent_pre + key_pre
            label = items["label_desc"]         #获取相应的标签
            example = [sent, label]             #形成[按字划分的句子，label]的组合，每一句对应一个组合？  
            self.append(example)

    def _numericalize(self, words, stoi):
        #数字化，讲对应的字转为对应id号，将汉字转为数字矢量表示
        #若不在stoi中，则可认为是被忽略的频数低于阈值的词，用UNK表示，在词典中对应为1
        return [1 if x not in stoi else stoi[x] for x in words]

    def numericalize(self, w2id, c2id):   #将汉字转换为对应的数字表示
        for i, example in enumerate(self):
            sent, label = example
            sent = self._numericalize(sent, w2id)    #将句子表示为矢量形式，每个字对应的下标代替该字来形成的向量
            label = c2id[label]                      #类别标记由desc转换为数字形式的类别名
            self[i] = (sent, label)


class DataBatchIterator(object):        #用于生成对应的迭代器
    def __init__(self, config, dataset="train",
                 is_train=True,         #默认为训练数据集
                 batch_size=32,         #默认batch_size为32
                 shuffle=False,         #无需将序列中的所有元素随机排序
                 batch_first=False,     #输入输出的第一维度不为batch_size
                 sort_in_batch=True):
        self.config = config            
        self.examples = DataSet(
            config=config, is_train=is_train, dataset=dataset)
        self.vocab = Vocab(config=config)   #词表
        self.cls_vocab = Vocab(specials=[], config=config)  #类别专用词表
        self.is_train = is_train
        self.max_seq_len = config.max_seq_len   #最大长度，可能不足的用<pad>来填充确保长度一致
        self.sort_in_batch = sort_in_batch
        self.is_shuffle = shuffle
        self.batch_first = batch_first  # [batch_size x seq_len x hidden_size]
        self.batch_size = batch_size
        self.num_batches = 0
        self.device = config.device

    def set_vocab(self, vocab):
        self.vocab = vocab

    def load(self, vocab_cache=None):  #将句子及标签读取出来并转为数字
        self.examples.read()     #将句子及类别描述简单地提取出来存储

        if not vocab_cache and self.is_train:
            # 0: 分过词的句子， 1: 关键词， 2: 标记
            self.vocab.make_vocab([x[0] for x in self.examples])  #形成句子的词表
            self.cls_vocab.make_vocab([[x[1]] for x in self.examples]) #形成类别描述的词表
            if not os.path.exists(self.config.save_vocab):#存储至文件下，后续直接读取即可
                torch.save(self.vocab, self.config.save_vocab + ".txt")
                torch.save(self.cls_vocab, self.config.save_vocab + ".cls.txt")
        else:  #非训练情况，则说明已经经过训练（验证与测试），故直接从文件中读取即可
            self.vocab = torch.load(self.config.save_vocab + ".txt")
            self.cls_vocab = torch.load(self.config.save_vocab + ".cls.txt")
        assert len(self.vocab) > 0 #断言，当条件表达式为false的时候触发异常
        self.examples.numericalize(   #将汉字数字矢量化，根据词表用索引下标替代该词
            w2id=self.vocab.stoi, c2id=self.cls_vocab.stoi)
        #向上取整，batches数为（句子总数/每个batch所含数量）
        self.num_batches = math.ceil(len(self.examples)/self.batch_size)

    def _pad(self, sentence, max_L, w2id, add_bos=False, add_eos=False):  #引入<pad>，使得每个句子形成的矢量长度一致=>矩阵
        if add_bos:    #是否引入句子的开头标识
            sentence = [w2id[BOS]] + sentence
        if add_eos:    #是否引入句子的结束标识
            sentence = sentence + [w2id[EOS]]
        if len(sentence) < max_L:  #确保每个句子形成的矢量相同，若不足则用<pad>补齐
            sentence = sentence + [w2id[PAD]] * (max_L-len(sentence))
        return [x for x in sentence]

    def pad_seq_pair(self, samples):   #取出统一长度后为batch.sent,label,mask赋值
        pairs = [pair for pair in samples]  #取出 [句子， 类别描述]
 
        Ls = [len(pair[0])+2 for pair in pairs]  #将句子部分取出，并且+2=>为BOS与EOS预留两个位置

        max_Ls = max(Ls)    #取出其中的句子最大长度
        sent = [self._pad(  #整理为相同长度
            item[0], max_Ls, self.vocab.stoi, add_bos=True, add_eos=True) for item in pairs]
        label = [item[1] for item in pairs]  #取出类别描述
        batch = Batch()     #以Batch的形式存储
        #为batch对象赋值
        batch.sent = torch.LongTensor(sent).to(device=self.device)  #将句子转换为整型张量，并且搬至相应设备上
        batch.label = torch.LongTensor(label).to(device=self.device)
        
        if not self.batch_first:  #如果未要求batch数最为第一维度时，则需转换句子的行列维度（存储形式）
            batch.sent = batch.sent.transpose(1, 0).contiguous()  #为tensonr转置后分配连续分布的内存空间
        batch.mask = batch.sent.data.clone().ne(0).long().to(device=self.device) #将sent部分克隆一份并将非0位置1，0位保持为0
        #[ne(0) => 不等于0]
        return batch

    def __iter__(self):   #划分为num_batches份数据，且确保等长 + 数字编码
        if self.is_shuffle:    #是否将元素变为乱序，在本场景下没有必要
            random.shuffle(self.examples)
        total_num = len(self.examples)  #总共存有的多少句子，样本
        for i in range(self.num_batches):  #将样本分成num_batches份batch_size数量的样本
            samples = self.examples[i * self.batch_size:
                                    min(total_num, self.batch_size*(i+1))]
            if self.sort_in_batch:    #按降序排列，主要是将长度相近的作为同一batch中，避免补充过多的<pad>
                samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
            yield self.pad_seq_pair(samples)   #同样以占用较低内存的方式生成迭代器

class Batch(object):
    def __init__(self):
        self.sent = None
        self.label = None
        self.mask = None
