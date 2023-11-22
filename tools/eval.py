import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report, hamming_loss, jaccard_score, multilabel_confusion_matrix

import os
import numpy as np
import pandas as pd

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
    # vocab_size = 72 # new
    # d_model = 256   # 隠れ層の次元数（256, 512, 1024）
    # nhead = 8   # Attention head (4, 8)
    # num_layers = 6  # エンコーダ層の数(3〜6)
    num_classes = 8 # 8ラベル分類
    # batch_size = 128    # バッチサイズ（16, 32, 64, 128）
    # learning_rate = 0.0001  # 学習率（0.0001, 0.001, 0.01, 0.1）
    # num_epochs = 10    # epoch数
    max_seq_length = 206    # 最大入力長
    # max_seq_length = 220    # new
    data_path = './data/hayazashi_Naotti.txt'
    checkpoint_dir = './checkpoints/'

    texts, labels = get_data(data_path)
    eval_texts = texts[2500:]
    eval_labels = labels[2500:]

    # データローダーのセットアップ
    eval_dataset = GeisterDataset(eval_texts, eval_labels, vocab_size, max_seq_length)
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
        p = prediction.to('cpu').detach().numpy().copy()
        # print(f"Predicted: {prediction}, Ground Truth: {ground_truth}")
        # print(p.argsort()[::-1])
        # print(p.argsort()[:3:-1])
        # print(f"Predicted: {prediction}, Ground Truth: {ground_truth}")
        pre = np.zeros(8)
        for red in p.argsort()[:3:-1]:
            pre[red] = 1
        # print(f"Predicted: {pre}, Ground Truth: {ground_truth}")


    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # 多ラベル混同行列
    conf_matrix = multilabel_confusion_matrix(np.array(all_ground_truth), np.array(all_predictions))

    # 適合率、再現率、F1値、サポート数
    report = classification_report(np.array(all_ground_truth), np.array(all_predictions), target_names=class_names)

    # ハミング損失
    hamming_loss_value = hamming_loss(np.array(all_ground_truth), np.array(all_predictions))

    # ジャッカード類似度
    jaccard_similarity = jaccard_score(np.array(all_ground_truth), np.array(all_predictions), average='samples')

    print("多ラベル混同行列:\n", conf_matrix)
    print("分類レポート:\n", report)
    print("ハミング損失:", hamming_loss_value)
    print("ジャッカード類似度:", jaccard_similarity)

    # 多ラベル混同行列
    # conf_matrix_df = pd.DataFrame(conf_matrix, columns=['TN', 'FP', 'FN', 'TP'])
    # 各行列をDataFrameに変換
    # conf_matrix_dfs = [pd.DataFrame(matrix.reshape((2, 2)), columns=['Predicted 0', 'Predicted 1', 'Actual 0', 'Actual 1']) for matrix in conf_matrix]
    # conf_matrix_dfs = [pd.DataFrame(matrix, columns=['Predicted 0', 'Predicted 1', 'Actual 0', 'Actual 1']) for matrix in conf_matrix]
    conf_matrix_dfs = [pd.DataFrame(matrix.reshape(1, -1), columns=['Predicted 0', 'Predicted 1', 'Actual 0', 'Actual 1']) for matrix in conf_matrix]
    # 各DataFrameを連結
    conf_matrix_df = pd.concat(conf_matrix_dfs, keys=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7'])
    conf_matrix_df.to_csv('confusion_matrix.csv', index=False)

    # 分類レポート
    report_df = pd.DataFrame.from_dict(classification_report(np.array(all_ground_truth), np.array(all_predictions), target_names=class_names, output_dict=True))
    report_df.to_csv('classification_report.csv')

    # ハミング損失とジャッカード類似度
    metrics_df = pd.DataFrame({'Hamming Loss': [hamming_loss_value], 'Jaccard Similarity': [jaccard_similarity]})
    metrics_df.to_csv('metrics.csv', index=False)

def main():
    d_model_list= [64, 128, 256]
    nhead_list = [4, 8]
    num_layers_list = [3, 4, 5, 6]
    batch_size_list = [16, 32, 64, 128, 256]
    # learning_rate_list = [0.0001, 0.001, 0.01, 0.1] 
    learning_rate_list = [0.00001, 0.0001, 0.001] 
    # for ln in learning_rate_list:
    #     for epoch in range(10):
    #         print(f'---epoch: {epoch+1}---')
    #         eval(256, 8, 6, 16, ln, epoch+1)

    for epoch in range(10):
        print(f'---epoch: {epoch+1}---')
        eval(256, 8, 6, 16, 0.0001, epoch+1)
    # eval(256, 8, 6, 16, 0.001, 10)
    # for d_model in d_model_list:
    #     for nhead in nhead_list:
    #         for num_layers in num_layers_list:
    #             for batch_size in batch_size_list:
    #                 for learning_rate in learning_rate_list:
    #                     for epoch in range(10):
    #                         eval(d_model, nhead, num_layers, batch_size, learning_rate, epoch+1)


if __name__ == '__main__':
    main()
