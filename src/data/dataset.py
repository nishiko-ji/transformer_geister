import torch
from torch.utils.data import Dataset, DataLoader
from src.data.tokenizer import GeisterTokenizer


class GeisterDataset(Dataset):
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
        tokenizer.select_vocab(1)
        text_tokens = tokenizer.tokenize(text, self.max_seq_length) # トークン化
        # print(text_tokens)
        text_tokens = tokenizer.convert_tokens_to_ids(text_tokens)    # IDに変換

        # ラベルを赤駒：1、青駒：0で表現
        # label2 = []
        # for l in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']:
        #     if l in label:
        #         label2.append(1.0)
        #     else:
        #         label2.append(0.0)

        # new
        # print(f'label: {label}')
        label2 = []
        u_num = 0
        r_num = 0
        b_num = 0
        for l in label.split(','):
            if l=='r':
                r_num += 1
            elif l=='b':
                b_num += 1
            else:
                u_num += 1

        for l in label.split(','):
            if l=='r':
                label2.append(1.0)
            elif l=='b':
                label2.append(0.0)
            else:
                label2.append((4-r_num)/u_num)

        # print(f'text_tolens: {text_tokens}')
        # print(f'label2: {label2}')

        return torch.LongTensor(text_tokens), torch.FloatTensor(label2)


if __name__ == '__main__':
    print()
