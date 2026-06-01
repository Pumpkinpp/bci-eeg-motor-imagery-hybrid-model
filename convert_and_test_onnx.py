import tensorflow as tf
import tf2onnx
import onnxruntime as ort
import numpy as np
import time
import os
import shutil

# ใช้พาธ Python จาก environment bci-project โดยตรงเพื่อเรียกโมดูล
PYTHON_EXE = r"C:\Users\palapeem\miniconda3\envs\bci-project\python.exe"
MODEL_PATH = "bci_hybrid_model.h5"
SAVED_MODEL_DIR = "temp_saved_model"
ONNX_MODEL_PATH = "bci_hybrid_model.onnx"

print("1. Loading Keras Model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("2. Converting Keras Model directly to ONNX...")
import tf2onnx
input_signature = [tf.TensorSpec([None, 64, 256, 1], tf.float32, name='input_layer')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

print(f"3. Saving ONNX model to {ONNX_MODEL_PATH}...")
with open(ONNX_MODEL_PATH, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"SUCCESS: ONNX model saved to {ONNX_MODEL_PATH}")

print("4. Testing Inference Comparison...")
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# ปรับ Input Shape ให้เข้ากับร่างยักษ์ Hybrid EEGNet-LSTM (64 Channels, 256 Timepoints, 1)
X_test_samples = np.random.randn(5, 64, 256, 1).astype(np.float32)

keras_times = []
onnx_times = []

# Warmup
_ = model.predict(X_test_samples[0:1], verbose=0)
_ = session.run(None, {input_name: X_test_samples[0:1]})

print("="*60)
print(f"{'Sample #':<10} | {'Keras Time (ms)':<20} | {'ONNX Time (ms)':<20}")
print("-"*60)

for i in range(5):
    sample = X_test_samples[i:i+1]
    
    start = time.perf_counter()
    _ = model.predict(sample, verbose=0)
    k_time = (time.perf_counter() - start) * 1000
    keras_times.append(k_time)
    
    start = time.perf_counter()
    _ = session.run(None, {input_name: sample})
    o_time = (time.perf_counter() - start) * 1000
    onnx_times.append(o_time)
    
    print(f"Sample {i+1:<3} | {k_time:<20.2f} | {o_time:<20.2f}")

print("="*60)
avg_k = np.mean(keras_times)
avg_o = np.mean(onnx_times)
print(f"AVERAGE    | {avg_k:<20.2f} | {avg_o:<20.2f}")
print(f"SPEEDUP    : {avg_k/avg_o:.2f}x faster with ONNX")
print("="*60)
