# train.py
import torch
from sklearn import metrics
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def assess(logit, target):
    # logit: n, output_size
    # target: n
    # target_names: output_size
    pred = torch.argmax(logit, dim=1)
    if torch.cuda.is_available():
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
    acc = accuracy_score(target, pred)
    p = precision_score(target, pred, average='macro')
    r = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')
    return acc, p, r, f1

def evaluation(outputs, labels):
    # correct -> 返回每个batch中分类正确的数量
    pred_y = torch.max(outputs, 1)[1].cuda().data.squeeze()
    # print("Predicts: ", pred_y)
    # print("Labels: ", labels)
    correct = torch.sum(torch.eq(pred_y, labels)).item()
    return correct

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()  # 將model的模式设为train，这样optimizer就可以更新model的参数
    criterion = nn.CrossEntropyLoss()  # 损失函数
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # 将模型参数传入optimizer，并设置learning rate
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        # training
        for i, (inputs, masks, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            masks = masks.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)  # device为"cuda"，將inputs转成torch.cuda.LongTensor，使用GPU进行计算
            optimizer.zero_grad()  # 由于loss.backward()的gradient会累加，所以每次喂完一个batch后需要清零
            outputs = model(inputs, masks)  # 將input喂给模型
            outputs = outputs.squeeze()  # 去掉最外面的dimension，将outputs用作criterion()
            loss = criterion(outputs, labels)  # 计算此时模型的training loss
            loss.backward()  # 反向传播，计算loss的gradient
            optimizer.step()  # 更新训练模型的参数
            correct = evaluation(outputs, labels)  # 计算此时模型的training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            if i % 10 == 0:
                print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
                    epoch + 1, i + 1, t_batch, loss.item(), correct * 100 / batch_size), end='\n')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))

        # validation
        model.eval()  # 将model的模式设置为eval，这样model的参数就会固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, masks, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                masks = masks.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)  # device为"cuda"，將inputs转成torch.cuda.LongTensor，使用GPU进行计算
                outputs = model(inputs, masks)  # 將input喂给模型
                loss = criterion(outputs, labels)  # 计算此时模型的training loss
                correct = evaluation(outputs, labels)  # 计算此时模型的validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch * 100))
            if total_acc > best_acc:
                # 如果validation的结果优于之前所有的結果，就把当下的模型保存
                best_acc = total_acc
                torch.save(model.state_dict(), "{}/model.pt".format(model_dir))
                # torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc / v_batch * 100))
                # model = torch.load("./tweets-original/ckpt.model")
                # model = model.to(device)  # device為"cuda"，GPU
                # test_loss, acc, p, r, f1 = eval_model(model, device, valid_iter=valid)
                # print('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f' % (
                # test_loss, acc, p, r, f1))

        print('-----------------------------------------------')
        model.train()  # 将model的模式重新设置回train，准备进行下一个batch的训练

def eval_model(model, batch_size, valid_iter, device):
    # validation
    criterion = nn.CrossEntropyLoss()  # 损失函数
    v_batch = len(valid_iter)
    model.eval()  # 将model的模式设置为eval，这样model的参数就会固定住
    with torch.no_grad():
        total, total_loss, total_acc = 0, 0, 0
        for i, (inputs, masks, labels) in enumerate(valid_iter):
            inputs = inputs.to(device, dtype=torch.long)
            masks = masks.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)  # device为"cuda"，將inputs转成torch.cuda.LongTensor，使用GPU进行计算
            outputs = model(inputs, masks)  # 將input喂给模型
            loss = criterion(outputs, labels)  # 计算此时模型的training loss
            correct = evaluation(outputs, labels)  # 计算此时模型的validation accuracy
            total_acc += correct / batch_size
            total_loss += loss.item()

        print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch))