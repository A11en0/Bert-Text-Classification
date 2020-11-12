# train.py
import torch
from torch import nn
import torch.optim as optim
import metrics
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("logs/SST-1")

# def assess(logit, target):
#     # logit: n, output_size
#     # target: n
#     # target_names: output_size
#     pred = torch.argmax(logit, dim=1)
#     if torch.cuda.is_available():
#         pred = pred.cpu().numpy()
#         target = target.cpu().numpy()
#     acc = accuracy_score(target, pred)
#     p = precision_score(target, pred, average='macro')
#     r = recall_score(target, pred, average='macro')
#     f1 = f1_score(target, pred, average='macro')
#     return acc, p, r, f1

def evaluation(outputs, labels):
    # correct -> 返回每个batch中分类正确的数量
    pred_y = torch.max(outputs, 1)[1].cuda().data.squeeze()
    # print("Predicts: ", pred_y)
    # print("Labels: ", labels)
    correct = torch.sum(torch.eq(pred_y, labels)).item()
    return correct

# def train(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
#     # total = sum(p.numel() for p in model.parameters())
#     # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     # print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
#     total_loss, total_acc, best_acc, best_score = 0, 0, 0, 0
#     test_loss, test_acc, test_p, test_r, test_f1 = 0, 0, 0, 0, 0
#
#     for epoch in range(n_epoch):
#         train_loss, eval_loss, acc, p, r, f1 = train_model(model, train, valid, batch_size, epoch, lr, device)
#         print('Epoch:%d, Training Loss:%.4f' % (epoch, train_loss))
#         print('Epoch:%d, Eval Loss:%.4f, Eval Acc:%.4f, Eval P:%.4f, Eval R:%.4f, Eval F1:%.4f' % (epoch, eval_loss,
#                     acc, p, r, f1))
#
#         if f1 > best_score:
#             best_score = f1
#             torch.save(model.state_dict(), 'results/%d_%s_%s.pt' % (epoch, 'Model', str(best_score)))
#             test_loss, test_acc, test_p, test_r, test_f1 = eval_model(model, valid, batch_size, device)
#
#         print('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f\n' % (test_loss, test_acc, test_p, test_r, test_f1))

def train_model(model, train_iter, valid_iter, batch_size, epoch, lr, device):
    model.train()  # 將model的模式设为train，这样optimizer就可以更新model的参数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # 将模型参数传入optimizer，并设置learning rate
    criterion = nn.CrossEntropyLoss()  # 损失函数
    total_loss, total_acc, best_acc = 0, 0, 0
    ind = 0
    for idx, (inputs, masks, labels) in enumerate(train_iter):
        # eval_acc, test_acc, train_loss, dev_loss, train_epoch = [], [], [], [], []
        inputs = inputs.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long)  # device为"cuda"，將inputs转成torch.cuda.LongTensor，使用GPU进行计算
        optimizer.zero_grad()  # 由于loss.backward()的gradient会累加，所以每次喂完一个batch后需要清零
        outputs = model(inputs, masks)  # 將input喂给模型
        outputs = outputs.squeeze()  # 去掉最外面的dimension，将outputs用作criterion()
        loss = criterion(outputs, labels)  # 计算此时模型的training loss
        loss.backward()  # 反向传播，计算loss的gradient
        optimizer.step()  # 更新训练模型的参数
        # correct = evaluation(outputs, labels)  # 计算此时模型的training accuracy
        # total_acc += (correct / batch_size)
        # print(len(train_iter) * epoch + ind)

        # writer.add_scalar("Loss/train", loss.item(),  len(train_iter) * epoch + ind)
        # writer.flush()

        if idx % 10 == 0:
            print('Epoch:%d, Idx:%d, Training Loss:%.4f' % (epoch, idx, loss.item()))

        total_loss += loss.item()
        ind += 1

    eval_loss, acc, p, r, f1 = eval_model(model, valid_iter, batch_size, device)
    return total_loss / ind, eval_loss, acc, p, r, f1

def eval_model(model, valid_iter, batch_size, device):
    # validation
    # v_batch = len(valid_iter)
    total, total_loss, total_acc = 0, 0, 0
    criterion = nn.CrossEntropyLoss()  # 损失函数
    model.eval()  # 将model的模式设置为eval，这样model的参数就会固定住
    with torch.no_grad():
        for i, (inputs, masks, labels) in enumerate(valid_iter):
            inputs = inputs.to(device, dtype=torch.long)
            masks = masks.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)  # device为"cuda"，將inputs转成torch.cuda.LongTensor，使用GPU进行计算
            outputs = model(inputs, masks)  # 將input喂给模型
            loss = criterion(outputs, labels)  # 计算此时模型的training loss
            # correct = evaluation(outputs, labels)  # 计算此时模型的validation accuracy
            # total_acc += correct / batch_size
            total_loss += loss.item()
        # print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch))

    acc, p, r, f1 = metrics.assess(outputs, labels)
    return total_loss / batch_size, acc, p, r, f1