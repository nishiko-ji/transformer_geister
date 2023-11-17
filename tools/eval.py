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


def eval(d_model, nhead, num_layers, batch_size, learning_rate, epoch):
    # torch.cuda.init()
    vocab_size = 84 # 語彙数
    # d_model = 256   # 隠れ層の次元数（256, 512, 1024）
    # nhead = 8   # Attention head (4, 8)
    # num_layers = 6  # エンコーダ層の数(3〜6)
    num_classes = 8 # 8ラベル分類
    # batch_size = 128    # バッチサイズ（16, 32, 64, 128）
    # learning_rate = 0.0001  # 学習率（0.0001, 0.001, 0.01, 0.1）
    num_epochs = 10    # epoch数
    max_seq_length = 206    # 最大入力長
    data_path = './data/hayazashi_Naotti.txt'
    checkpoint_dir = './checkpoints/'

    texts, labels = get_data(data_path)

    # データローダーのセットアップ
    eval_dataset = GeisterDataset(texts, labels, vocab_size, max_seq_length)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    # モデルの設定
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('cuDNNの有効状態：', torch.backends.cudnn.enabled)

    model = TransformerMultiClassClassifier(vocab_size, d_model, nhead, num_layers, num_classes).to(device)

    checkpoint_path = checkpoint_dir + f'ckpt_dmodel_{d_model}_nhead_{nhead}_numlayers_{num_layers}_batch_{batch_size}_rate_{str(learning_rate)[2:]}_epoch_{epoch}.pt'
    model.load_state_dict(torch.load(checkpoint_path))
    
    model.eval()
    # 予測と正解のリスト
    all_predictions = []
    all_ground_truth = []
    all_outputs = []

    with torch.no_grad():
        for batch_input, batch_labels in eval_loader:
            batch_input, batch_labels = batch_input.to(device), batch_labels.to(device)
            # モデルの順伝播（forward）で予測を取得
            outputs = model(batch_input)
            all_outputs.extend(outputs)
            predictions = (outputs > 0.5).long()  # しきい値0.5を使用してバイナリ予測に変換
            all_predictions.extend(predictions.cpu().numpy())
            all_ground_truth.extend(batch_labels.cpu().numpy())

    # 予測と正解を表示
    # for prediction, ground_truth in zip(all_predictions, all_ground_truth):
    #     print(f"Predicted: {prediction}, Ground Truth: {ground_truth}")
    for prediction, ground_truth in zip(all_outputs, all_ground_truth):
        print(f"Predicted: {prediction}, Ground Truth: {ground_truth}")
    

def main():
    d_model_list= [64, 128, 256]
    nhead_list = [4, 8]
    num_layers_list = [3, 4, 5, 6]
    batch_size_list = [16, 32, 64, 128, 256]
    learning_rate_list = [0.0001, 0.001, 0.01, 0.1] 
    eval(64, 4, 3, 16, 0.0001, 10)

    # for d_model in d_model_list:
    #     for nhead in nhead_list:
    #         for num_layers in num_layers_list:
    #             for batch_size in batch_size_list:
    #                 for learning_rate in learning_rate_list:
    #                     for epoch in range(10):
    #                         eval(d_model, nhead, num_layers, batch_size, learning_rate, epoch+1)


if __name__ == '__main__':
    main()
