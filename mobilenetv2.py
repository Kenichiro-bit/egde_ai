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
    results = predict_mobilenet_v2("image.png")
    for cls, name, prob in results:
        print(f"{name}: {prob:.3f}")
