import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import os

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
            

def train():
    # torch.cuda.init()
    vocab_size = 84
    d_model = 256
    nhead = 8
    num_layers = 6
    num_classes = 8
    batch_size = 128
    learning_rate = 0.0001
    num_epochs = 100
    max_seq_length = 206
    data_path = ''
    checkpoint_dir = ''

    texts, labels = get_data(data_path)

    # データローダーのセットアップ
    train_dataset = GeisterDataset(texts, labels, vocab_size, max_seq_length)
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
        for batch_input, batch_labels in train_loader:
            batch_input, batch_labels = batch_input.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % save_interval == 0:
            checkpoint_filename = os.path.join(checkpoint_dir, f'model_checkpoint_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_filename)
            print(f'Saved checkpoint at epoch {epoch + 1} to {checkpoint_filename}')

def main():
    train()


if __name__ == '__main__':
    main()
