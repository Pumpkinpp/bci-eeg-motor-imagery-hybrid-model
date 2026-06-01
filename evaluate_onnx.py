import numpy as np
import onnxruntime as rt
from sklearn.metrics import classification_report

# Load ONNX model
onnx_model_path = "bci_hybrid_model.onnx"
sess = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

# Load raw test data (let's take 1000 samples from the end as 'test set' for report)
try:
    X_full = np.load("X_raw.npy")
    y_full = np.load("y_raw.npy")
    # Take last 1000 samples
    X_test = X_full[-1000:].astype(np.float32)
    y_test = y_full[-1000:]
except Exception as e:
    print("Error loading data:", e)
    exit()

print("Running ONNX inference for Classification Report...")
# Run prediction
pred_probs = sess.run(None, {input_name: X_test})[0]
y_pred = np.argmax(pred_probs, axis=1)

# Generate classification report
target_names = ['Left Hand', 'Right Hand', 'Resting Mode']
report = classification_report(y_test, y_pred, target_names=target_names)
print("==================================================")
print("[CLASSIFICATION REPORT (ONNX MODEL)]")
print("==================================================")
print(report)
print("==================================================")
