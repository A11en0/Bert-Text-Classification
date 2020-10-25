import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from gensim.models import word2vec, Word2Vec
import transformers as ppb
from torchtext import data
from torchtext.vocab import Vectors

# 实现dataset所需要的'__init__', '__getitem__', '__len__'
# 为dataloader做准备
class TextDataset(data.Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)

    __len__ will return the number of data
    """

    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

# preprocess.py
class Preprocess1():
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        # 读取训练好的word2vec模型
        self.embedding = Word2Vec.load(self.w2v_path)
        # self.embedding = gensim.models.KeyedVectors.load_word2vec_format(self.w2v_path, binary=True)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # 把word加入embedding，并赋予随机生成的representation vector
        # word只會是"<PAD>"或"<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得训练好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # word2idx -> dictionary
        # idx2word -> list
        # word2vector -> list
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i + 1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 将"<PAD>"和"<UNK>"加入embedding
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # 把句子里面的字转成想对应的index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i + 1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)

class Preprocess():
    def __init__(self, sentences):
        # self.w2v_path = w2v_path
        self.sentences = sentences
        # self.sen_len = sen_len
        # self.idx2word = []
        # self.word2idx = {}
        # self.embedding_matrix = []
        # self.tokenize = []

        self.model_class, self.tokenizer_class, self.pretrained_weights = (
        ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    # def get_bert(self):
    #     model = self.model_class.from_pretrained(self.pretrained_weights)
    #     return model

    def pad_sequence(self, sentence):
        max_len = 0
        for i in sentence.values:
            if len(i) > max_len:
                max_len = len(i)
        padded = np.array([i + [0] * (max_len - len(i)) for i in sentence.values])
        return padded

    def sentence_word2idx(self):
        # 把句子里面的字转成想对应的index
        tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        tokenized = self.sentences.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        padded = self.pad_sequence(tokenized)
        attention_mask = np.where(padded != 0, 1, 0)
        return (torch.LongTensor(padded), torch.LongTensor(attention_mask))

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)

def load_data(path, label, pos):
    '''
    用于加载数据集
    '''
    with open(path, 'r', encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
        text = [line.split("\t")[pos].strip("\n").split(" ") for line in lines]
        label = [label for line in lines]
    return text, label

def load_dataset(train_data_path):
    df = pd.DataFrame()
    with open(train_data_path, 'r', encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
        texts = [line.split()[:-1] for line in lines]
        # text = [line.split("\t")[pos].strip("\n").split(" ") for line in lines]
        # label = [line.split()[-1].replace("\n", "") for line in lines]
        labels = [line.split()[-1] for line in lines]
    df['text'] = texts
    df['name'] = labels

    labelMap = {elem: index for index, elem in enumerate(set(df["name"]))}
    df['label'] = df['name'].map(labelMap)
    return df

def train_test_split(all_iter, ratio):
    length = len(all_iter)
    train_data = []
    test_data = []
    train_start = 0
    train_end = int(length*ratio)
    ind = 0
    for batch in all_iter:
        if ind < train_end:
            train_data.append(batch)
        else:
            test_data.append(batch)
        ind += 1
    return train_data, test_data

def train_dev_split(train_iter, ratio):
    length = len(train_iter)
    train_data = []
    dev_data = []
    train_start = 0
    train_end = int(length*ratio)
    ind = 0
    for batch in train_iter:
        if ind < train_end:
            train_data.append(batch)
        else:
            dev_data.append(batch)
        ind += 1
    return train_data, dev_data

if __name__ == "__main__":
    # 定义数据存放位置， labels, texts列表临时保存读取的数据
    path_prefix = '/home/allen/Code/python/Datasets/tweets-original/'

    # labels, texts = [], []
    # # 加载所有数据集，并做拼接
    # x1, y1 = load_data(path_prefix+"chelsea-raw.txt", "chelsea", 7)
    # x2, y2 = load_data(path_prefix+"obama-raw.txt", "obama", 6)
    # x3, y3 = load_data(path_prefix+"smartphone-raw.txt", "smartphone", 7)
    # x4, y4 = load_data(path_prefix+"blackfriday-raw.txt", "blackfriday", 7)
    # x5, y5 = load_data(path_prefix+"arsenal-raw.txt", "arsenal", 7)
    # texts+=x1+x2+x3+x4+x5
    # labels+=y1+y2+y3+y4+y5

    textDF = load_dataset("./datasets/sst-2.tsv")

    preprocess = Preprocess(textDF["text"])
    input_ids, attention_mask = preprocess.sentence_word2idx()
    label = preprocess.labels_to_tensor(textDF["label"])


