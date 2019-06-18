# Sensor Signal With Python
センサーで関連(活動分類とか)のサンプルコードを管理する。
ここでの成果物

    - Dataset Collections  (dev)
    - Dataset Separator    (dev)

などは後々独立したものにする予定。

## System Requirements
- Python 3.6.8
- PyTorch 1.1

## `Sensor Dataset Collections`
機械学習分野においてセンサーデータを使ったコードの数が画像分類や自然言語処理よりも
少ない原因の一つに有名なデータセットが無いことが原因であると考える。
そこで、ここではMNISTやCIFARのように手軽にデータセットを準備できるような環境を準備する。


`Sensor Dataset Collections` は次のようなインターフェースでデータの入手・利用ができるようにする。


```python
from sd_collections.uci import load_har
from sd_collections.utils import dataset_separator
X, y = load_har(raw_data=True)

(x_train, y_train), (x_test, y_test) = dataset_separator(X, y)

```


