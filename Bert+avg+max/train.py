from model import *
from data_loader import *
import time
from torch.nn import CrossEntropyLoss

def train_model(model, device, train_loader, dev_loader):
    #定义优化器
    #mylog = open('result.log', mode = 'a',encoding='utf-8')
    param_optimizer = list(model.named_parameters())  # 模型参数名字列表
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    #optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,          #仅对模型参数做较小的修改
                     warmup=0.05,
                     t_total=len(train_loader) * NUM_EPOCHS)
    model.train()
    criterion = CrossEntropyLoss(reduction="sum")  #定义损失函数
    best_acc = 0.0
    #开始训练
    for epoch in range(1, NUM_EPOCHS + 1):
        best_acc = 0.0
        start_time = time.time()
        for batch_idx, (x1,x2,x3, y) in enumerate(train_loader):
            x1,x2,x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            y_pred = model([x1, x2, x3])  # 得到预测结果
            model.zero_grad()             # 梯度清零
            loss = criterion(y_pred, y.squeeze())  # 得到loss
            loss.backward()
            optimizer.step()
            if(batch_idx + 1) % 200 == 0:    # 打印loss
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(x1), 
                      len(train_loader.dataset),
                      100. * batch_idx / len(train_loader), 
                      loss.item()))  # 记得为loss.item()atches, valid_loss))
        end_time = time.time()
        print("time: {}".format(end_time - start_time))
        acc = valid_model(model, device, dev_loader)  #每训练完一个epoch,参数在验证集中的表现情况
        if best_acc < acc: 
            best_acc = acc 
            torch.save(model.state_dict(), PATH)  # 保存最优模型
            print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc)) 



def valid_model(model, device, dev_loader):
    model.eval()
    test_loss = 0.0 
    acc = 0 
    criterion = CrossEntropyLoss(reduction="sum")
    for batch_idx, (x1,x2,x3, y) in enumerate(dev_loader):
        x1,x2,x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model([x1,x2,x3])
        test_loss += criterion(y_, y.squeeze())
        pred = y_.max(-1, keepdim=True)[1]   # .max(): 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()    # 记得加item()
    test_loss /= len(dev_loader)
    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
          test_loss, acc, len(dev_loader.dataset),
          100. * acc / len(dev_loader.dataset)))
    return acc / len(dev_loader.dataset)


def main():
    #构建训练集的dataloader
    #加载数据集至对应的列表中
    data_set(path + "train.json", train=True)
    data_set(path + "dev.json", dev=True)
    train_data = TensorDataset(torch.LongTensor(input_ids_train), 
                           torch.LongTensor(input_types_train), 
                           torch.LongTensor(input_masks_train), 
                           torch.LongTensor(label_train))
    train_sampler = RandomSampler(train_data)  
    train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    #构建验证集的dataloader
    dev_data = TensorDataset(torch.LongTensor(input_ids_dev), 
                          torch.LongTensor(input_types_dev), 
                         torch.LongTensor(input_masks_dev),
                          torch.LongTensor(label_dev))
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)
    
    #开始训练
    model = Model().to(DEVICE)
    train_model(model, DEVICE, train_loader, dev_loader)
    
if __name__ == "__main__":
    main()
