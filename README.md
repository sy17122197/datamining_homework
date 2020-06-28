# datamining_homework
sy's homework    基于神经网络模型的新闻多分类应用

介绍
模型介绍：
CNN、LSTM、WordAVG、Bert

环境：
python3.7
pytorch1.2.0
sklearn

数据集：
数据集在tnews_public目录下，目录下有四个文件分别对应与测试集、验证集、训练集以及类别说明
test.json
dev.json
train.json
labels.json

训练过程以及模型测试结果记录：
存于每个文件下的result.txt文档中，前面为训练的损失函数变化情况；
最后一行记录相应的测试集模型得分

预训练语言模型：
bert预训练模型放在chinese_wwm_ext_pytorch目录下，目录下有三个文件：
·bert_config.json
·pytorch_model.bin
·vocab.txt
也可以自行下载：
   下载地址：https://github.com/ymcui/Chinese-BERT-wwm
