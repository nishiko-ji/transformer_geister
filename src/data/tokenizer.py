import pprint


class GeisterTokenizer():
    """ ガイスター用Tokenizer
    
    自然言語処理モデルでガイスターの学習をするためのTokenizer

    Attributes:
        vocab (dict[str, int]): トークンとIDの辞書

    """
    def __init__(self):
        """ 初期化

        初期化処理としてvocabの作成を行う。

        """
        self.vocab = self.make_vocab()

    def make_vocab(self) -> dict[str, int]:
        """ vocab(トークンとIDの辞書)を作成

        トークン(str)とID(int)の辞書で語彙を表現する。

        Returns:
            dict[str, int]: トークンとIDの辞書(vocab)

        """
        special_tokens = ['<sos>', '<eos>', '<pad>', '<unk>']   # 特殊トークンリスト
        # special_tokens = ['<sos>', '<eos>', '<sep>', '<pad>', '<unk>']   # 特殊トークンリスト
        token_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']   # コマ名リスト
        # token_list = ['r', 'b', 'u']   # new
        # 棋譜のトークン化
        for player in ['0', '1']:
            for koma in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                for direction in ['E', 'W', 'S', 'N']:
                    token_list.append(f'{player}{koma}{direction}')
        vocab = {}
        # 辞書化
        for i, data in enumerate(special_tokens+token_list):
            vocab[data] = i

        return vocab
    
    def tokenize(self, text: str, max_length: int) -> list[str]:
        """ トークン化

        入力文からトークン配列を作成する。

        Args:
            text (str): 入力文
            max_length (int): 最大入力長

        Returns:
            list[str]: トークン配列

        """
        tokens = [token for token in text.split(',')]   # textを','で分割
        tokens = tokens[:max_length - 2]    # 最大入力長までトリミング
        tokens += ['<pad>'] * (max_length - len(tokens) - 2)  # 最大入力長までパディング
        tokens.insert(0, '<sos>')   # 開始トークンを挿入
        tokens.append('<eos>')  # 終了トークンを挿入

        return tokens

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """ トークン配列からID配列に変換

        辞書self.vocabを利用してトークン配列からID配列に変換する。

        Args:
            tokens (list[str]): トークン配列

        Returns:
            list[int]: ID配列

        """
        if self.vocab is None:  # vocabが空なら
            raise ValueError("Vocabulary is not set.")

        return [self.vocab[token] for token in tokens]  # 変換して返す

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """ ID配列からトークン配列に変換

        辞書self.vocabを利用してID配列からトークン配列に変換する。

        Args:
            ids (list[int]): ID配列

        Returns:
            list[str]: トークン配列

        """
        if self.vocab is None:  # vocabが空なら
            raise ValueError("Vocabulary is not set.")

        return [key for key, value in self.vocab.items() if value in ids]   # 変換して返す

    def print_vocab(self):
        """ 語彙のトークンとIDのセットを表示
        
        vocabに保存されているトークンとIDのセットを表示する。

        """
        pprint.pprint(sorted(self.vocab.items(), key=lambda i: i[1]))   # IDでソートして整形表示


if __name__ == '__main__':
    tokenizer = GeisterTokenizer()  
    tokenizer.print_vocab()
    # tokens = tokenizer.tokenize('A,D,F,G,0AN,1AN,0AN,1DE,0AN,1CE,0BW,1AN,0CW,1AW,0CN,1AN,0DN,1AN,0HE,1CN,0CE,1CN,0BN,1BW,0DN,1BW,0DN,1EN,0EN,1BN,0FN,1EW,0CE,1DN,0EW,1FN,0CS,1BN,0EE,1FN,0GN,1FN,0ES,1BS,0FN,1FW,0CS,1FN,0CN,1GN,0FN,1BN,0FE,1BS,0FN,1BN,0FN,1BS,0FE,1BN,0FE,1ES,0BN,1DN,0BN,1DN,0BS,1GN,0BN,1DN,0BN,1DE', 200)
    # print(tokens)
    # print(tokenizer.convert_tokens_to_ids(tokens))
    # print(len(tokenizer.convert_tokens_to_ids(tokens)))

