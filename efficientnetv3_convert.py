import tensorflow as tf

# 1. EfficientNetB0 モデルを直接ロード
model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=True
)

# 2. TFLite コンバータの作成
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# サイズ最適化を有効化（量子化は代表データセットが必要ですが、OPTIMIZE_DEFAULT だけでも一部最適化されます）
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 3. （必要なら）代表データセットを用意
# def representative_data_gen():
#     for input_value in calibration_inputs:
#         yield [input_value]
# converter.representative_dataset = representative_data_gen

# 4. 変換実行
tflite_model = converter.convert()

# 5. ファイルに保存
with open("efficientnet_b0_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite モデルを出力しました: efficientnet_b0_int8.tflite")
