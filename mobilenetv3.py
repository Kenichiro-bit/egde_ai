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
    for name, description, score in predict_mobilenet_v3("image.png"):
        print(f"{description}: {score:.3f}")
