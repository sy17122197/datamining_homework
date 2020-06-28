import torch
import torch.nn as nn
from config import parse_config
from data_loader import DataBatchIterator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def test_textrnn_model(model,  test_data, criterion, config):
    # Build optimizer.
    # params = [p for k, p in model.named_parameters(
    # ) if p.requires_grad and "embed" not in k]
    model.eval()   
    #使用eval()函数，让model变成测试模式
    #dropout和batch normalization的操作在训练和测试的模式下是不一样的。
    total_loss = 0.
    total_precision = 0.
    total_recall = 0.
    total_f1 = 0.
    total_acc = 0.
    test_data_iter = iter(test_data)
    for idx, batch in enumerate(test_data_iter):
        ground_truth_g = batch.label
        with torch.no_grad():
            outputs = model(batch.sent)
        # probs = model.generator(decoder_outputs)
        loss = criterion(outputs, ground_truth_g)  #得到损失函数值
        total_loss += loss   #计算总体损失值
        pred_g = outputs.max(-1, keepdim=True)[1].squeeze(1)
        ground_truth = ground_truth_g.cpu()
        pred = pred_g.cpu()
        total_precision += precision_score(ground_truth, pred, average='macro')
        total_recall += recall_score(ground_truth, pred, average='macro')
        total_f1 += f1_score(ground_truth, pred, average='macro')
        total_acc += accuracy_score(ground_truth, pred)
        num = idx + 1
    return total_loss/num, total_precision/num, total_recall/num, total_f1/num, total_acc /num

def main():
    # 读配置文件
    config = parse_config()
    # 载入测试集合
    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        batch_size=config.batch_size,
        shuffle=True)
    test_data.load()
    
    # 载入textcnn模型
    model = torch.load("results/model.pt")
    #print(model)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    # Do training.
    loss, precision, recall, f1, acc= test_textrnn_model(model,  test_data, criterion, config)
    print("test loss: {0:.2f},  precision: {1:.2f},  recall:{2:.2f},  f1:{3:.2f}, acc:{4:.2f}".format(
            loss, precision, recall, f1, acc))



if __name__ == "__main__":
    main()