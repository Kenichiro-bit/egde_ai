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
