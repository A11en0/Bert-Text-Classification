from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from model import LSTM_Net, Bert_Net, BiLSTM
from utils import *
# from sklearn.model_selection import train_test_split
from utils import train_dev_split, train_test_split
from train_eval import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs/SST-1")

def main():
    dataset = "sst-2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 48
    epochs = 50
    lr = 2e-6
    # load_model = "results/36_Model_0.746031746031746.pt"
    load_model = ""

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

    # model = Bert_Net(embedding_dim=768, hidden_dim=300, num_layers=4, dropout=0.5)
    model = BiLSTM(embedding_dim=768, hidden_dim=300)
    # model1 = LSTM_Net(embedding, embedding_dim=768, hidden_dim=300, num_layers=2, dropout=0.5,
    #              fix_embedding=True, embedding_pretrained_path="models/model.pt")

    model = model.to(device)  # device為"cuda"，GPU

    # if train_mode:
    #     print("--- Start Training --- ")
    #     training(batch_size, epoch, lr, model_dir, train_data, dev_data, model, device)
    # else:
    #     print("--- Start Testing --- ")
    #     model.load_state_dict(torch.load("./models/model.pt"))
    #     eval_model(model, batch_size, test_data, device)

    # train(batch_size, epoch, lr, model_dir, train_data, dev_data, model, device)

    if load_model:
        model.load_state_dict(torch.load(load_model))
        test_loss, acc, p, r, f1 = eval_model(model, test_data, batch_size, device)
        print('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f' % (test_loss, acc, p, r, f1))
        return

    best_score = 0.0
    test_loss, test_acc, test_p, test_r, test_f1 = 0, 0, 0, 0, 0

    for epoch in range(epochs):
        train_loss, eval_loss, acc, p, r, f1 = train_model(model, train_data, dev_data, batch_size, epoch, lr, device)
        print('Epoch:%d, Training Loss:%.4f' % (epoch, train_loss))
        print('Epoch:%d, Eval Loss:%.4f, Eval Acc:%.4f, Eval P:%.4f, Eval R:%.4f, Eval F1:%.4f' % (epoch, eval_loss,
                    acc, p, r, f1))

        if f1 > best_score:
            best_score = f1
            torch.save(model.state_dict(), 'results/%d_%s_%s.pt' % (epoch, 'Model', str(best_score)))
            test_loss, test_acc, test_p, test_r, test_f1 = eval_model(model, test_data, batch_size, device)

        print('Test Loss:%.4f, Test Acc:%.4f, Test P:%.4f, Test R:%.4f, Test F1:%.4f\n'
              % (test_loss, test_acc, test_p, test_r, test_f1))

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/eval", eval_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        writer.add_scalar('Accuracy/eval', acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.flush()

    writer.close()
if __name__ == '__main__':
    main()