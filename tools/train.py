import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset import GeisterDataset
from src.model.transformer import TransformerMultiClassClassifier


def get_data(path):
    texts = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            data = line.rstrip('\n').split(' ')
            red_pos0 = ','.join(data[0])
            texts.append(f'{red_pos0},{data[2]}')
            red_pos1 = ','.join(data[1])
            labels.append(red_pos1)

    return texts, labels

def get_data2(path):
    texts = []
    labels = []
    with open(path, 'r') as f:
        for line in f:
            data = line.rstrip('\n').split(' ')
            red_pos0 = data[0]
            texts.append(f'{red_pos0},{data[2]}')
            red_pos1 = data[1]
            labels.append(red_pos1)

    return texts, labels


def train(d_model, nhead, num_layers, batch_size, learning_rate):
    # torch.cuda.init()
    vocab_size = 84 # 語彙数
    # vocab_size = 72 # new
    # d_model = 256   # 隠れ層の次元数（256, 512, 1024）
    # nhead = 8   # Attention head (4, 8)
    # num_layers = 6  # エンコーダ層の数(3〜6)
    num_classes = 8 # 8ラベル分類
    # batch_size = 128    # バッチサイズ（16, 32, 64, 128）
    # learning_rate = 0.0001  # 学習率（0.0001, 0.001, 0.01, 0.1）
    num_epochs = 10    # epoch数
    # max_seq_length = 206    # 最大入力長
    max_seq_length = 220    # new
    data_path = './data/hayazashi_Naotti.txt'
    checkpoint_dir = './checkpoints/'

    texts, labels = get_data(data_path)

    # データローダーのセットアップ
    train_texts = texts[:2500]
    train_labels = labels[:2500]
    train_dataset = GeisterDataset(train_texts, train_labels, vocab_size, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # モデルの設定
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('cuDNNの有効状態：', torch.backends.cudnn.enabled)

    model = TransformerMultiClassClassifier(vocab_size, d_model, nhead, num_layers, num_classes).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_interval = 1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        for batch_input, batch_labels in train_loader:
            batch_input, batch_labels = batch_input.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        average_train_loss = train_loss / num_batches
        print(f'Epoch {epoch +1}, Average Training Loss: {average_train_loss}')

        if (epoch + 1) % save_interval == 0:
            checkpoint_filename = os.path.join(checkpoint_dir, f'ckpt_dmodel_{d_model}_nhead_{nhead}_numlayers_{num_layers}_batch_{batch_size}_rate_{str(learning_rate)[2:]}_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_filename)
            print(f'Saved checkpoint at epoch {epoch + 1} to {checkpoint_filename}')

def main():
    # d_model_list= [64, 128, 256]
    # nhead_list = [4, 8]
    # num_layers_list = [3, 4, 5, 6]
    # batch_size_list = [16, 32, 64, 128, 256]
    learning_rate_list = [0.00001, 0.0001, 0.001] 
    d_model_list= [256]
    nhead_list = [8]
    num_layers_list = [6]
    batch_size_list = [16]
    # learning_rate_list = [0.0001] 

    for d_model in d_model_list:
        for nhead in nhead_list:
            for num_layers in num_layers_list:
                for batch_size in batch_size_list:
                    for learning_rate in learning_rate_list:
                        train(d_model, nhead, num_layers, batch_size, learning_rate)


if __name__ == '__main__':
    main()
