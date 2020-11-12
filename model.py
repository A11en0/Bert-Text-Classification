import torch
from torch import nn
import transformers as ppb
from torch.autograd import Variable
from transformers import DistilBertModel

class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.5, output_size=2):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.W = nn.Linear(hidden_dim * 2, embedding_dim, bias=False)
        self.b = nn.Parameter(torch.ones([embedding_dim]))

    def forward(self, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1*2, len(X), self.hidden_dim)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), self.hidden_dim)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model

class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5,
                 fix_embedding=True):

        super(LSTM_Net, self).__init__()
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否将 embedding fix住，如果fix_embedding为False，在训练过程中，embedding也会跟着被训练
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, 5),
                                         nn.Softmax() )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最后一层的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

class Bert_Net(nn.Module):
    # def __init__(self, bert_config, tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1, use_cuda=False):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.5, output_size=2):
        super(Bert_Net, self).__init__()
        self.pretrained_weights = 'distilbert-base-uncased'
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        # self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        # self.liner = nn.Linear(hidden_dim*2, tagset_size+2)

        # self.model_class, self.tokenizer_class, self.pretrained_weights = (
        # ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
        # self.word_embeds = BertModel.from_pretrained(bert_config)

        self.word_embeds = DistilBertModel.from_pretrained(self.pretrained_weights)

        self.gama = 0.5
        da = hidden_dim
        db = int(da/2)

        for param in self.word_embeds.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # self.W1 = nn.Linear(2 * hidden_dim + embedding_dim,
        #                     da)  # (da, 2u+d) => (hidden_size, embedding_dim+2*hidden_size)
        # self.w1 = nn.Linear(da, 1, bias=False)
        # self.W2 = nn.Linear(embedding_dim, db)
        # self.w2 = nn.Linear(db, 1, bias=False)
        # self.output = nn.Linear(2 * hidden_dim + embedding_dim, output_size)

        self.classifier = nn.Sequential(self.dropout,
                                        nn.Linear(hidden_dim, 2),
                                        nn.Softmax())

    # def self_attention(self, H):
    #     # H: batch_size, seq_len, 2*hidden_size
    #     hidden_size = H.size()[-1]
    #     Q = H
    #     K = H
    #     V = H
    #     atten_weight = F.softmax(torch.bmm(Q, K.permute(0, 2, 1))/math.sqrt(hidden_size), -1) # batch_size, seq_len, seq_len
    #     A = torch.bmm(atten_weight, V) # batch_size, seq_len, 2*hidden_size
    #     A = A.permute(0, 2, 1) # batch_size, 2*hidden_size, seq_len
    #     # q: short text representation
    #     q = F.max_pool1d(A, A.size()[2]).squeeze(-1) # batch_size, 2*hidden_size ==> (128, 128, 1).squeeze(-1) -> (128, 128)
    #     return q
    #
    # def cst_attention(self, c, q):
    #     # c: batch_size, concept_seq_len, embedding_dim
    #     # q: batch_size, 2*hidden_size
    #     # print(q.size())
    #     # print(c.size())
    #     q = q.unsqueeze(1)
    #     q = q.expand(q.size(0), c.size(1), q.size(2))
    #     c_q = torch.cat((c, q), -1) # batch_size, concept_seq_len, embedding_dim+2*hidden_size
    #     c_q = self.w1(F.tanh(self.W1(c_q))) # batch_size, concept_seq_len, 1
    #     alpha = F.softmax(c_q.squeeze(-1), -1) # batch_size, concept_seq_len
    #
    #     return alpha
    #
    # def ccs_attention(self, c):
    #     # c: batch_size, concept_seq_len, embedding_dim
    #     c = self.w2(F.tanh(self.W2(c))) # batch_size, concept_seq_len, 1
    #     beta = F.softmax(c.squeeze(-1), -1) # batch_size, concept_seq_len
    #
    #     return beta

    def forward(self, sentence, attention_mask=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state
        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''

        embeds = self.word_embeds(sentence, attention_mask=attention_mask)[0]
        out, hidden = self.lstm(embeds)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

    # def forward(self, sentence, attention_mask=None):
    #     '''
    #     args:
    #         sentence (word_seq_len, batch_size) : word-level representation of sentence
    #         hidden: initial hidden state
    #     return:
    #         crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
    #     '''
    #     # batch_size = sentence.size(0)
    #     # seq_length = sentence.size(1)
    #
    #     embeds = self.word_embeds(sentence, attention_mask=attention_mask)[0]
    #     out, hidden = self.lstm(embeds)
    #     out = out[:, -1, :]
    #     out = self.classifier(out)
    #     return out
