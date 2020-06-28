import torch
#模型中用到的参数
path = "tnews_public/"   #数据地址
bert_path = "chinese_wwm_ext_pytorch/"  #bert预训练模型地址
num_class = 15       #类别数
pad_size = 75       # max_len
BATCH_SIZE = 20
PATH = 'roberta_model.pth'    #训练好的模型保存地址
NUM_EPOCHS = 3                #训练的轮次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")