import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import GeisterTokenizer


class CustomDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[str], vocab_size: int, max_seq_length: int):
        self.texts = texts  # 入力文
        self.labels = labels    # 答えlabel
        self.vocab_size = vocab_size    # 語彙数
        self.max_seq_length = max_seq_length    # 入力の最大語数

    def __len__(self):
        return len(self.texts)  # データ数

    def __getitem__(self, idx: int):
        text = self.texts[idx]  # idx番目のtext
        label = self.labels[idx]    # idx番目のlabel

        tokenizer = GeisterTokenizer()  # トークナイザ
        text_tokens = tokenizer.tokenize(text, self.max_seq_length) # トークン化
        tokenizer.convert_tokens_to_ids(text_tokens)    # IDに変換

        # ラベルを赤駒：1、青駒：0で表現
        label2 = []
        for l in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
            if l in label:
                label2.append(1.0)
            else:
                label2.append(0.0)

        return torch.LongTensor(text_tokens), torch.FloatTensor(label2)


if __name__ == '__main__':
    print()
