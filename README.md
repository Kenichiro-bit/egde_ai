# egde_ai
make edge ai tool

以下に、フェーズ1–2で挙げた各モデルの「PC上での実装」用サンプルコード例を示します。いずれも「推論用」の最小構成で、必要に応じてデータローダーや前処理を組み込んでください。

---

### 1. MobileNet V2 （TensorFlow / Keras）

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# モデル読み込み
model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=True
)
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
decode_preds = tf.keras.applications.mobilenet_v2.decode_predictions

def predict_mobilenet_v2(img_path: str):
    img = Image.open(img_path).resize((224, 224))
    x = preprocess(np.array(img)[None, ...])
    preds = model(x)
    return decode_preds(preds.numpy(), top=3)[0]

if __name__ == "__main__":
    results = predict_mobilenet_v2("test.jpg")
    for cls, name, prob in results:
        print(f"{name}: {prob:.3f}")
```

---

### 2. MobileNet V3-Small （TensorFlow / Keras）

```python
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.applications.MobileNetV3Small(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=True
)
preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
decode_preds = tf.keras.applications.mobilenet_v3.decode_predictions

def predict_mobilenet_v3(img_path: str):
    img = Image.open(img_path).resize((224, 224))
    x = preprocess(np.array(img)[None, ...])
    preds = model(x)
    return decode_preds(preds.numpy(), top=3)[0]

if __name__ == "__main__":
    for name, description, score in predict_mobilenet_v3("test.jpg"):
        print(f"{description}: {score:.3f}")
```

---

### 3. EfficientNet-Lite0 （TensorFlow Lite）

#### 3.1 モデル変換（Keras → TFLite）

```bash
# 事前に mobilenet_v3_small.h5 や EfficientNet-Lite0 を Kerasで用意しておく
tflite_convert \
  --keras_model_file=efficientnet_lite0.h5 \
  --output_file=efficientnet_lite0.tflite \
  --optimizations=OPTIMIZE_FOR_SIZE \
  --representative_dataset=calib.py
```

#### 3.2 推論コード（tflite-runtime）

```python
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# インタプリタ初期化
interpreter = tflite.Interpreter(model_path="efficientnet_lite0.tflite")
interpreter.allocate_tensors()

# 入力／出力テンソル情報
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_efficientnet_lite0(img_path: str):
    img = Image.open(img_path).resize(
        (input_details[0]['shape'][2], input_details[0]['shape'][1])
    )
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds[0]  # 後処理(decoding)は必要に応じて

if __name__ == "__main__":
    scores = predict_efficientnet_lite0("test.jpg")
    top_idx = scores.argsort()[-3:][::-1]
    print("Top-3 indices:", top_idx, "scores:", scores[top_idx])
```

---

### 4. ResNet-18 （PyTorch）

```python
import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image

# モデル読み込み
model = resnet18(pretrained=True)
model.eval()

# 前処理
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225])
])

# 推論関数
def predict_resnet18(img_path: str):
    img = Image.open(img_path)
    x = transform(img).unsqueeze(0)  # バッチ次元
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    top3 = probs.topk(3)
    return top3.indices.tolist(), top3.values.tolist()

if __name__ == "__main__":
    idxs, vals = predict_resnet18("test.jpg")
    for idx, val in zip(idxs, vals):
        print(f"class {idx}: {val:.3f}")
```

---

### 5. Custom CNN（PyTorch）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import DataLoader

# シンプルCNN定義
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 56 * 56, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 112×112
        x = self.pool(F.relu(self.conv2(x)))  # 56×56
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# データセット＆ローダー例 (ImageFolder 構成)
transform = Resize((224,224)), ToTensor()
train_ds = ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

# 学習ループの雛形
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(train_ds.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# 推論例
model.eval()
sample, _ = next(iter(train_loader))
with torch.no_grad():
    preds = model(sample.to(device))
    print(torch.argmax(preds, dim=1)[:5])
```

---

このコード群で、各モデルの「読み込み → 前処理 → 推論（または学習）」の流れが掴めるはずです。

* **MobileNet / EfficientNet** はまずKeras/TFLiteでワンライナーで動かせる手軽さが魅力。
* **ResNet-18** は PyTorch 標準。
* **Custom CNN** は構造を自由に変えられるので、最初に学習＆デバッグ用に最適です。

次は、これらをベンチマークして「速度 vs 精度」の表にまとめましょう。何かつまずいた点があればお気軽に！


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
このエラーは、保存した HDF5 ファイルに含まれるレイヤー設定の一部（ここではおそらく `batch_input_shape=(None, 224, 224, 3)` などの引数）が「位置引数」としてシリアライズされていて、`load_model` がそれを「キーワード引数で渡すべき」と解釈しており読み込めないことが原因です。

対処法は大きく２つあります。

---

## 1. `.h5` を経由せず、直接 Keras アプリケーション API でモデルを生成する

最も簡単で確実なのが、そもそも `.h5` をロードせずに以下のように直接モデルをインスタンス化してしまう方法です。TF が持っている公式の MobileNetV3Small を呼び出せば、必ず動く状態のモデルオブジェクトが得られます。

```python
import tensorflow as tf

# HDF5 をロードせずに直接生成
model = tf.keras.applications.MobileNetV3Small(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=True
)

# 以下はそのまま TFLiteConverter に渡せます
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("mobilenet_v3_small_int8.tflite", "wb") as f:
    f.write(tflite_model)
```

──これなら余計なシリアライズ・デシリアライズの問題を回避できます。

---

## 2. カスタムオブジェクトを指定して HDF5 をロードする

もしどうしてもファインチューニングした後の `.h5` を読み込みたい場合は、そのモデルを構成したレイヤーやユーティリティ関数を `custom_objects` に渡す必要があります。例えば、TensorFlow Hub 経由でロードしたレイヤーを使っていたり、独自の前処理レイヤーを挟んでいるケースです。

```python
import tensorflow as tf
import tensorflow_hub as hub

# 例: Hub Layer を使っている場合
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model = tf.keras.models.load_model(
    "mobilenet_v3_small_finetuned.h5",
    custom_objects=custom_objects
)
```

**ポイント**

* `ValueError: Only input tensors may be passed as positional arguments` は、モデル構成の一部を「位置引数」にしていた箇所を「キーワード引数」で再マッピングしてほしいという意味。
* 公式モデルをそのまま使うなら（方法①）、`load_model` を使わずに `tf.keras.applications` から呼び出すのが最速です。

---

### まとめ

* **公式 ImageNet モデルをそのまま変換** するなら、`.h5` ファイルを経由せずに `tf.keras.applications.MobileNetV3Small(...)` でモデルを作り、直接 `TFLiteConverter.from_keras_model` に渡すのが一番簡単。
* **カスタムモデル（ファインチューニング済みなど）をロード** したい場合は、使用したすべてのカスタムレイヤー／関数を `load_model(..., custom_objects={…})` で指定して読込みましょう。

まずは①の方法で動かしてみて、「自分でファインチューニングしたモデル」を変換したい段階になったら②の方法で `.h5` をロードしてみてください。うまくいったらお知らせください！
