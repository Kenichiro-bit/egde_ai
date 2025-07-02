ベンチマーク（精度＋速度）のためだけに１０００クラスすべてを使う必要はありません。評価用の **代表的なサブセット** を使うことで、

* **評価コスト** （データ準備・推論時間）が下がり
* **開発ループ** （「このモデルは速い？遅い？」の判断）が回りやすく

なります。以下、２つのアプローチを紹介します。

---

## A. 既製のサブセットを使う

1. **Imagenette（10 クラス）**

   * ImageNet の人気クラスを 10 種まで絞ったデータセット
   * ダウンロード・セットアップが簡単（先ほどの `download_and_extract_archive` で対応）
   * **推論速度** と **精度傾向** を把握するのに十分

2. **ImageNet-100**

   * ImageNet のうち任意の 100 クラスをピックアップしたサブセット
   * PyTorch 公式にはありませんが、公開リポジトリからラベル一覧を取得して `ImageFolder` で読み込めます
   * 「中規模クラス数」での精度感を見るのに最適

3. **Tiny-ImageNet（200 クラス）**

   * 64×64 画像サイズで 200 クラス、ImageNet の軽量版
   * モデル入力サイズを合わせるにはリサイズが必要ですが、クラス数は 200 程度に抑えられます

---

## B. ImageNet-100 を自作する方法

もし「ちょうど 100 クラス分だけ」で評価したい場合は、以下の流れで自分でサブセットを作れます。

1. **クラス ID リストを用意**

   ```python
   # 例: 100個の ImageNet クラス ID をテキストファイルに列挙しておく
   with open("imagenet100.txt") as f:
       selected = [line.strip() for line in f]
   ```
2. **tf.data でフィルタリング**（TensorFlow の場合）

   ```python
   import tensorflow as tf

   def filter_fn(image, label):
       # label は 0–999 の整数
       return tf.reduce_any(tf.equal(label, selected_indices))

   ds = tf.keras.preprocessing.image_dataset_from_directory(
       "imagenet/train",
       labels="inferred",
       label_mode="int",
       batch_size=32,
       image_size=(224,224)
   )
   ds100 = ds.filter(lambda x, y: filter_fn(x, y))
   ```
3. **PyTorch でフィルタリング**（torchvision の場合）

   ```python
   from torchvision.datasets import ImageFolder
   from torch.utils.data import Subset

   full_ds = ImageFolder("imagenet/train", transform=transform)
   # full_ds.classes → ['n01440764', 'n01443537', …] のリスト
   selected_classes = set(open("imagenet100.txt").read().splitlines())
   # インデックス化
   idxes = [i for i, (_, lbl) in enumerate(full_ds) if full_ds.classes[lbl] in selected_classes]
   ds100 = Subset(full_ds, idxes)
   ```
4. **DataLoader／tf.data に詰めて推論・精度評価**

   ```python
   loader100 = DataLoader(ds100, batch_size=32, shuffle=False, num_workers=4)
   # あとは同じベンチマーク用ループで評価
   ```

---

## おすすめの進め方

1. **最初は Imagenette（10 クラス）で高速に**

   * 画像サイズ 160×160 なので PC 上でもすぐ終わる
   * モデルごとの速度差・精度差が掴みやすい

2. **中盤は Tiny-ImageNet（200 クラス）でウォッチ**

   * もう少しクラス間の判別が難しくなるので、モデルの「本番感」もつかめる

3. **最終的に必要なら ImageNet-100（100 クラス）で本番評価**

   * ここまで来れば「1000 クラス全部で回すのと結果が大きく変わらない」ことが多い

---

### ベンチマークスクリプト例（擬似コード）

```python
import time

def benchmark(model_fn, dataloader):
    times, correct = [], 0
    total = 0
    for imgs, labels in dataloader:
        start = time.perf_counter()
        preds = model_fn(imgs)         # 画像バッチを渡して推論
        times.append(time.perf_counter() - start)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)
    print(f"Avg latency: {sum(times)/len(times)*1000:.1f} ms")
    print(f"Accuracy: {correct/total*100:.2f}%")
```

* `model_fn`：TFLite なら `lambda x: run_tflite(x)`、PyTorch/TensorFlow ならモデルの `__call__`
* `dataloader`：Imagenette / Tiny-ImageNet / ImageNet-100 のいずれか

---

**結論**：

* **いきなり 1000 クラスを回す必要はなし**。
* **まずは小〜中規模のサブセット**（10～200クラス）で差を掴み、
* **最終評価で 100 クラス or 1000 クラス**、という段階的アプローチがおすすめです。

これでステップ１の「クラス数の削減」についてクリアになったと思いますが、いかがでしょうか？続きを相談したい点があれば教えてください！


こちらは **Imagenette (10 クラス)** と **Tiny-ImageNet (200 クラス)** を自動ダウンロード＆展開し、`ImageFolder`＋`DataLoader` で読み込んだ上で、先ほどのベンチマーク関数を回すサンプルコードです。

```python
import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

# ベンチマーク関数（推論時間＆Top-1精度を測定）
def benchmark(model_fn, dataloader, device="cpu"):
    model_fn = model_fn.to(device)
    model_fn.eval()
    total, correct = 0, 0
    times = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            start = time.perf_counter()
            outputs = model_fn(imgs)
            times.append(time.perf_counter() - start)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_latency = sum(times) / len(times) * 1000  # ms
    accuracy   = correct / total * 100
    print(f"Avg latency: {avg_latency:.1f} ms  |  Accuracy: {accuracy:.2f}%")

# 前処理パイプライン
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── A-1: Imagenette (10 クラス) ─────────────────────────
imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
download_and_extract_archive(imagenette_url, download_root="data")
train_ds_10 = ImageFolder("data/imagenette2-160/train", transform=transform)
val_ds_10   = ImageFolder("data/imagenette2-160/val",   transform=transform)
val_loader_10 = DataLoader(val_ds_10, batch_size=32, shuffle=False, num_workers=4)

# モデル例：TFLite ではなく PyTorch の MobileNetV3
# （TFLite を使う場合は model_fn で interpreter.invoke() ラップ）
from torchvision.models import mobilenet_v3_small
model_10 = mobilenet_v3_small(pretrained=True)

print("=== Imagenette (10 クラス) ===")
benchmark(model_10, val_loader_10, device="cpu")


# ── A-2: Tiny-ImageNet (200 クラス) ────────────────────
tiny_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
download_and_extract_archive(tiny_url, download_root="data", filename="tiny-imagenet-200.zip")
# 展開後フォルダ: data/tiny-imagenet-200/train, val
train_ds_200 = ImageFolder("data/tiny-imagenet-200/train", transform=transform)
val_dir = "data/tiny-imagenet-200/val/images"  # Tiny-ImageNet の検証は少し構造が特殊
#  val/val_annotations.txt を読んで ImageFolder 用に symlink かコピーで準備しておく必要あり
#  ここでは簡単化のため train データでベンチ
val_ds_200   = train_ds_200  # 実際は検証セット用意をお願いします
val_loader_200 = DataLoader(val_ds_200, batch_size=32, shuffle=False, num_workers=4)

model_200 = mobilenet_v3_small(pretrained=True)
print("\n=== Tiny-ImageNet (200 クラス) ===")
benchmark(model_200, val_loader_200, device="cpu")
```

**ポイント**

* `download_and_extract_archive` で自動ダウンロード＆解凍。
* **Imagenette** はそのまま `ImageFolder` 構造になっているので簡単。
* **Tiny-ImageNet** は公式検証ディレクトリ構造が特殊なので、実際には `val_annotations.txt` を元に検証用フォルダを整形してから読み込むと精度評価も可能です。
* `model_fn` を TFLite 版に変えれば同じベンチマークができます（`mobilenet_v3_small` 部分を `run_tflite` 関数に差し替えてください）。

これをベースに、クラス数やモデルを変えつつ「速度 vs 精度」の比較を進めてみてください！
