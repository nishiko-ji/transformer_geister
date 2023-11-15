import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import GeisterTokenizer

class CustomDataset(Dataset):
    def __init__(self, texts, labels, vocab_size, max_seq_length):
        self.texts = texts  # 入力文
        self.labels = labels    # 答えlabel
        self.vocab_size = vocab_size    # 語彙数
        self.max_seq_length = max_seq_length    # 入力の最大語数

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # トークン化
        tokenizer = GeisterTokenizer()  
        text_tokens = tokenizer.tokenize(text, self.max_seq_length)
        tokenizer.convert_tokens_to_ids(text_tokens)

        # ラベルを4つのクラスに分類
        label2 = []
        for l in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            if l in label:
                label2.append(1.0)
            else:
                label2.append(0.0)

        return torch.LongTensor(text_tokens), torch.FloatTensor(label2)
