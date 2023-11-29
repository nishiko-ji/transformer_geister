# transformer_geister
実装中

Transformerを用いてガイスターの敵駒色推定を行う。

同じ相手と複数回対戦することで、相手の特徴を学習する。

相手の行動をすべて入力し、行動順序を考慮した駒推定を行う。


## 実装メモ
1. 最終局面の取得済駒を色確定
2. 最終手：敵駒ゴール（相手勝利）->ゴール駒は青
	     自駒ゴール（自分勝利）->未確定
	     相手駒を取得（相手勝利）->取得駒は赤
	     相手駒を取得（自分勝利）->取得駒は青
	     自駒が取られた（自分勝利）->未確定
   　もし、相手勝利なら
3. 色未確定駒の色確率を計算

pieces_listで駒色と位置を確認可能


texts: <sos>, A, B, C, D, E, F, G, H, <sep>, a, b, c, d, e, f, g, h, <sep>, ...., <eos>

       1    , 8                     , 1    , 8                     , 1    , 200 , 1

       sp   , r,b                   , sp   , r,b,u                 , sp   , move, sp

220

max_seq_length = 218

labels: a, b, c, d, e, f, g, h
