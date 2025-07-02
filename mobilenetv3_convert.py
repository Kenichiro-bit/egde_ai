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
