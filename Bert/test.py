import torch
from model import *
from data_loader import *
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def test_model(model, device, test_loader):    # 测试模型, 得到测试集评估结果
    criterion = CrossEntropyLoss(reduction="sum")  #定义损失函数
    model.eval()
    total_loss = 0.0 
    total_precision = 0.
    total_recall = 0.
    total_f1 = 0.
    total_acc = 0.
    for batch_idx, (x1,x2,x3, y) in enumerate(test_loader):
        x1,x2,x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        ground_truth_g = y.squeeze()
        with torch.no_grad():
            y_ = model([x1,x2,x3])
        total_loss += criterion(y_, y.squeeze())  # 得到loss
        pred_g = y_.max(-1, keepdim=True)[1]   # .max(): 2输出，分别为最大值和最大值的index
        ground_truth = ground_truth_g.cpu()
        pred = pred_g.cpu()
        total_precision += precision_score(ground_truth, pred, average='macro')
        total_recall += recall_score(ground_truth, pred, average='macro')
        total_f1 += f1_score(ground_truth, pred, average='macro')
        total_acc += accuracy_score(ground_truth, pred)
    test_loss = total_loss/len(test_loader)
    test_precision =  total_precision/len(test_loader)
    test_recall = total_recall/len(test_loader)
    test_f1 = total_f1/len(test_loader)
    test_acc = total_acc/len(test_loader)
    print('\nTest set: Average loss: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}, F1:{3:.4f}, acc:{4:4f}'.format(
          test_loss, test_precision, test_recall, test_f1, test_acc))
    return

def main():
    #加载数据集
    data_set(path + "test.json", test=True)
     #构建测试集的迭代器
    test_data = TensorDataset(torch.LongTensor(input_ids_test), 
                          torch.LongTensor(input_types_test), 
                         torch.LongTensor(input_masks_test),
                          torch.LongTensor(label_test))
    test_sampler = SequentialSampler(test_data)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
    #加载最优模型进行测试
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("roberta_model.pth"))
    #进行测试
    test_model(model, DEVICE, test_loader)
    
    


if __name__ == "__main__":
    main()