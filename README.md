# datamining_homework
sy's homework    基于神经网络模型的新闻多分类应用

#模型介绍：

CNN(有详细的注释)、LSTM、WordAVG、Bert

#环境：

python3.7、pytorch1.2.0、sklearn

#数据集：

数据集在tnews_public目录下，目录下有四个文件分别对应与测试集、验证集、训练集以及类别说明

test.json

dev.json

train.json

labels.json

#训练过程以及模型测试结果记录：

存于每个文件下的result.txt文档中，前面为训练的损失函数变化情况；

最后一行记录相应的测试集模型得分

#CNN、LSTM、WordAVG训练得到的模型

存于results目录下，并且命名为model.pt

#预训练语言模型：

Bert预训练模型由于上传大小限制，故给出链接

   下载地址：https://github.com/ymcui/Chinese-BERT-wwm
   
   选择：RoBERTa-wwm-ext, Chinese


基于Bert预训练模型得到句子编码直接放入线性层进行分类模型地址：

百度网盘：链接：https://pan.baidu.com/s/193PE6TTeBWvSepRiv2JllA      
提取码：yn7t


基于Bert预训练模型取中间编码层第六层进行取均值，最大值的模型地址：

百度网盘：链接：https://pan.baidu.com/s/1uLyGNdWIsNaNzTN3OhXovA     
提取码：56mt

