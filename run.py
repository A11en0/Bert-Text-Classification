from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from model import LSTM_Net, Bert_Net
from train_eval import training, eval_model
from utils import *
import os
import torch
import argparse
import numpy as np
from torch import nn
# from sklearn.model_selection import train_test_split
from utils import train_dev_split, train_test_split

# def old_main():
#     path_prefix = './tweets-original/'
#     labels, texts = [], []
#     print("--- Loading Data ---")
#     # 加载所有数据集，并做拼接
#     x1, y1 = load_data(path_prefix+"chelsea-raw.txt", "chelsea", 7)
#     x2, y2 = load_data(path_prefix+"obama-raw.txt", "obama", 6)
#     x3, y3 = load_data(path_prefix+"smartphone-raw.txt", "smartphone", 7)
#     x4, y4 = load_data(path_prefix+"blackfriday-raw.txt", "blackfriday", 7)
#     x5, y5 = load_data(path_prefix+"arsenal-raw.txt", "arsenal", 7)
#     texts+=x1+x2+x3+x4+x5
#     labels+=y1+y2+y3+y4+y5
#     # 使用labels, texts创建一个DataFrame保存数据
#     textDF = pd.DataFrame()
#     textDF['text'] = texts
#     textDF['name'] = labels
#
#     labels_set = list(set(textDF.name))
#     for label in labels_set:
#         print(label, textDF[textDF.name == label].text.count())
#
#     textDF['label'] = textDF['name'].apply(lambda x: labels_set.index(x))
#     data_for_use = pd.DataFrame()
#     for label in labels_set:
#         data_for_use = data_for_use.\
#             append(textDF[textDF.name == label].sample(10000, replace=False))
#     data_for_use.dropna(axis=0, how='any', inplace=True)
#
#     # GPU for CUDA computing
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # w2v_path = os.path.join(path_prefix, 'w2v_tweet.model')  # 處理word to vec model的路徑
#     # w2v_path = os.path.join(path_prefix, 'GoogleNews-vectors-negative300.bin') # 處理word to vec model的路徑
#
#     sen_len = 30
#     fix_embedding = True  # fix embedding during training
#     batch_size = 128
#     epoch = 50
#     lr = 0.0002
#     train_mode = True
#
#     # model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
#     model_dir = path_prefix  # model directory for checkpoint model
#
#     # print("loading data ...")  # 把'training_label.txt'跟'training_nolabel.txt'讀進來
#     # train_x, y = load_training_data("")
#     # train_x_no_label = load_training_data(train_no_label)
#
#     X_train, y_train = data_for_use.text, data_for_use.label
#     print("X_train shape: ", X_train.shape)
#     # preprocess = Preprocess(X_train, sen_len, w2v_path=w2v_path)
#
#     preprocess = Preprocess(X_train)
#     # embedding = preprocess.make_embedding(load=True)
#     input_ids, attention_mask = preprocess.sentence_word2idx()
#     y = preprocess.labels_to_tensor(y_train)
#
#     # 把data分成training data和validation data
#     X_train, X_test, y_train, y_test = train_test_split(input_ids, y, test_size = 0.33, random_state = 42) # X_train: (335, 768) X_test: (165, 768)
#
#     # 把data做成dataset供dataloader使用
#     train_dataset = TwitterDataset(X=X_train, y=y_train)
#     val_dataset = TwitterDataset(X=X_test, y=y_test)
#
#     # 把 data 转成 batch of tensors
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                num_workers=8)
#     val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              num_workers=8)
#
#     if train_mode:
#         print("--- Start Training --- ")
#         model = Bert_Net(embedding_dim=768, hidden_dim=300, num_layers=4, dropout=0.5)
#
#         # model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=300, num_layers=4, dropout=0.5,
#         #                  fix_embedding=fix_embedding)
#         model = model.to(device)  # device為"cuda"，GPU
#         training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)
#
#     else:
#         model = torch.load("./tweets-original/ckpt.model")
#         model = model.to(device)  # device為"cuda"，GPU
#         test_loss, acc, p, r, f1 = eval_model(model, device, val_loader)
#         print('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f' % (test_loss, acc, p, r, f1))
#
#     print(model)

def main():
    dataset = "sst-2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 48
    epoch = 50
    lr = 1e-5
    train_mode = False
    model_dir = "./models"  # model directory for checkpoint model

    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    textDF = load_dataset("./datasets/" + dataset + ".tsv")
    preprocess = Preprocess(textDF["text"])
    # input_ids, attention_mask = preprocess.sentence_word2idx()
    input_ids, attention_mask = preprocess.sentence_word2idx()
    labels = preprocess.labels_to_tensor(textDF["label"])

    train_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8)

    # 把data做成dataset供dataloader使用
    # train_dataset = TextDataset(X=input_ids, y=labels)
    # 把 data 转成 batch of tensors
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            batch_size=batch_size,
    #                                            shuffle=True,
    #                                            num_workers=8)

    train_data, test_data = train_test_split(train_loader, 0.8)
    train_data, dev_data = train_dev_split(train_data, 0.8)

    model = Bert_Net(embedding_dim=768, hidden_dim=300, num_layers=4, dropout=0.5)
    model = model.to(device)  # device為"cuda"，GPU

    if train_mode:
        print("--- Start Training --- ")
        training(batch_size, epoch, lr, model_dir, train_data, dev_data, model, device)
    else:
        print("--- Start Testing --- ")
        model.load_state_dict(torch.load("./models/model.pt"))
        eval_model(model, batch_size, test_data, device)

if __name__ == '__main__':
    main()