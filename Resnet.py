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
    idxs, vals = predict_resnet18("image.png")
    for idx, val in zip(idxs, vals):
        print(f"class {idx}: {val:.3f}")
