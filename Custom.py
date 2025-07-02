import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torchvision import transforms

# 1) 前処理パイプラインを Compose で定義
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 2) データをダウンロード＆展開（最初だけ実行）
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
download_and_extract_archive(url, download_root="data")

# 3) ImageFolder でデータセット読み込み
train_ds = ImageFolder("data/imagenette2-160/train", transform=transform)
val_ds   = ImageFolder("data/imagenette2-160/val",   transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

# 4) シンプルCNN 定義
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 56 * 56, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # →112×112
        x = self.pool(F.relu(self.conv2(x)))  # →56×56
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 5) トレーニングループ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(train_ds.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}  Loss: {total_loss/len(train_loader):.4f}")

# 6) 推論サンプル
model.eval()
sample_imgs, _ = next(iter(val_loader))
with torch.no_grad():
    preds = model(sample_imgs.to(device))
    print("予測クラス：", torch.argmax(preds, dim=1)[:5].cpu().tolist())
