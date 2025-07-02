# egde_ai
make edge ai tool


「事前に～を Keras で用意しておく」というのは、TFLite 変換ツール（`tflite_convert` や Python API）に渡す **Keras のモデルファイル（HDF5 形式）** をあらかじめ手元に用意しておく、という意味です。具体的には次のいずれかを行います。

---

## A. 公式のプリトレイン済みモデルをそのまま保存する

### 1. MobileNet V3 Small を HDF5 に保存

```python
import tensorflow as tf

# ImageNet 重み付きの MobileNetV3 Small をロード
model = tf.keras.applications.MobileNetV3Small(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=True
)

# HDF5 形式で保存
model.save('mobilenet_v3_small.h5')
print("mobilenet_v3_small.h5 を出力しました")
```

これで `mobilenet_v3_small.h5` がカレントフォルダにできます。

### 2. EfficientNet-Lite0 を HDF5 に保存

TensorFlow 2.10 以降では `tf.keras.applications` に EfficientNet-Lite 系が含まれています。もしお使いのバージョンにない場合は、TensorFlow Hub から取り込む方法もあります。

#### a. `tf.keras.applications` がサポートしている場合

```python
from tensorflow.keras.applications import EfficientNetLite0

model = EfficientNetLite0(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=True
)
model.save('efficientnet_lite0.h5')
print("efficientnet_lite0.h5 を出力しました")
```

#### b. TensorFlow Hub から取り込む場合

```bash
pip install tensorflow-hub
```

```python
import tensorflow as tf
import tensorflow_hub as hub

# TF-Hub の EfficientNet-Lite0 feature-vector モジュール
hub_url = "https://tfhub.dev/google/efficientnet/lite0/feature-vector/2"
base = hub.KerasLayer(hub_url, input_shape=(224,224,3))
model = tf.keras.Sequential([
    base,
    tf.keras.layers.Dense(1000, activation='softmax')  # ImageNet クラス数
])
model.save('efficientnet_lite0.h5')
print("efficientnet_lite0.h5 を出力しました")
```

---

## B. カスタムモデルを訓練して保存する場合

もし独自データで再学習（ファインチューニング）したい場合は、

1. 上記のプリトレイン済みモデルを `model.fit(...)` で再学習
2. 学習後に同じく `model.save('your_model.h5')` で保存

```python
# 例: MobileNetV3Small をファインチューニング
base = tf.keras.applications.MobileNetV3Small(
    weights='imagenet', include_top=False, input_shape=(224,224,3)
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dense(10, activation='softmax')(x)  # クラス数に合わせる
model = tf.keras.Model(base.input, x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(train_dataset, validation_data=val_dataset, epochs=3)
model.save('mobilenet_v3_small_finetuned.h5')
```

---

これで `.h5` ファイルが用意できれば、先に示した `tflite_convert` コマンドや Python API に渡して TFLite モデルへ変換できます。

> **まとめ**
>
> * **プリトレイン済みモデル** をそのまま保存 → すぐ変換
> * **ファインチューニング後** に保存 → データセットに最適化されたモデルを変換
> * 保存形式は Keras の **HDF5 (`.h5`)**、あるいは SavedModel 形式でも OK（`--saved_model_dir` を使用）。

何か不明点があればまた教えてください！
