import time
import numpy as np
import onnxruntime as rt
import warnings
warnings.filterwarnings('ignore')

print("==================================================")
print("[BCI REAL-TIME INFERENCE SYSTEM (ONNX)]")
print("==================================================")

# 1. โหลดโมเดล ONNX เพื่อความเร็วสูงสุด
onnx_model_path = "bci_hybrid_model.onnx"
print(f"[1] Loading ONNX Model: {onnx_model_path}...")
sess = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

# 2. โหลดข้อมูลจำลองสัญญาณสมองจริง (Real Data Simulation)
print("[2] Connecting to EEG Data Stream...")
try:
    X_stream = np.load("X_raw.npy")
    y_stream = np.load("y_raw.npy")
    # สุ่มเลือกข้อมูล 5 จังหวะ (Epochs) เพื่อจำลองการรับข้อมูลจริง
    indices = [100, 500, 1500, 3000, 5000]
    print("[SUCCESS] Data Stream Connected. Ready for real-time inference.\n")
except FileNotFoundError:
    print("[ERROR] X_raw.npy or y_raw.npy not found.")
    exit()

# แมปตัวเลขคลาสเป็นคำสั่งควบคุมเกม
class_mapping = {0: "Left Hand", 1: "Right Hand", 2: "Resting Mode"}

print("==================================================")
print("[STARTING REAL-TIME PREDICTION]")
print("==================================================")

total_inference_time = 0
num_samples = len(indices)

for i, idx in enumerate(indices):
    # ดึงข้อมูล 1 Sample (ขนาด 1 x 64 x 256 x 1) เหมือนรับเข้ามา 1 วินาที
    single_eeg_epoch = X_stream[idx:idx+1].astype(np.float32)
    actual_label = y_stream[idx]
    
    # จับเวลาและทำนายผล
    start_time = time.perf_counter()
    pred_probs = sess.run(None, {input_name: single_eeg_epoch})[0]
    end_time = time.perf_counter()
    
    pred_class = np.argmax(pred_probs)
    confidence = pred_probs[0][pred_class] * 100
    infer_time_ms = (end_time - start_time) * 1000
    total_inference_time += infer_time_ms
    
    print(f"[Stream Packet #{i+1}] Length: 256 samples | Channels: 64")
    print(f"   - Inference Time : {infer_time_ms:.2f} ms")
    print(f"   - Prediction     : {class_mapping[pred_class]} (Confidence: {confidence:.2f}%)")
    print(f"   - Actual Command : {class_mapping[actual_label]}")
    
    if pred_class == actual_label:
        print("   -> [MATCHED]")
    else:
        print("   -> [MISMATCHED]")
    print("-" * 50)
    
    # จำลองหน่วงเวลาเหมือนรอรับข้อมูลใหม่แบบ Real-time
    time.sleep(0.5)

print("\n[INFERENCE SUMMARY]")
print(f"Average Inference Speed: {total_inference_time/num_samples:.2f} ms per sample")
print("Status: READY FOR UNITY GAME DEPLOYMENT")
print("==================================================")
