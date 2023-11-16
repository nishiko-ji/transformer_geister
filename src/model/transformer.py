import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Transformerモデルの構築
class TransformerMultiClassClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, num_classes):
        super(TransformerMultiClassClassifier, self).__init__()

        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformerエンコーダ層
        self.encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)

        # 出力層
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # 埋め込み層
        x = self.transformer_encoder(x)  # Transformerエンコーダ
        x = x.mean(dim=1)  # シーケンスの平均
        logits = self.fc(x)  # 出力層
        # return logits
        return torch.sigmoid(logits)
