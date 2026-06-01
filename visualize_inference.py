import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as rt
import time
import os

# --- Configuration ---
ONNX_MODEL_PATH = "bci_hybrid_model.onnx"
DATA_X_PATH = "X_raw.npy"
DATA_Y_PATH = "y_raw.npy"
NUM_SAMPLES_TO_PLOT = 15  # จำนวนช่วงสัญญาณที่จะพล็อต
CHANNEL_TO_PLOT = 0       # เลือก Channel ที่จะแสดง (เช่น 0 คือ Channel แรก)
CLASS_LABELS = ["Left Hand", "Right Hand", "Resting"]
COLORS = ['#FF5555', '#55FF55', '#5555FF'] # Red, Green, Blue

def visualize_inference():
    print("==================================================")
    print("[BCI INFERENCE VISUALIZATION SYSTEM]")
    print("==================================================")

    # 1. Load Model
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"[ERROR] Model file {ONNX_MODEL_PATH} not found.")
        return
    
    sess = rt.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    # 2. Load Data
    print("[1] Loading data for visualization...")
    try:
        X_raw = np.load(DATA_X_PATH)
        y_raw = np.load(DATA_Y_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # เลือกช่วงข้อมูลมาแสดง (สุ่มเลือกช่วงที่ต่อเนื่องกันเล็กน้อยหรือกระจาย)
    # ในที่นี้เลือก 15 samples ต่อเนื่องจากจุดเริ่มต้นที่สุ่ม
    start_idx = np.random.randint(0, len(X_raw) - NUM_SAMPLES_TO_PLOT)
    indices = range(start_idx, start_idx + NUM_SAMPLES_TO_PLOT)

    all_signals = []
    all_probs = []
    all_preds = []
    all_actuals = []

    print(f"[2] Running inference on {NUM_SAMPLES_TO_PLOT} segments...")
    for idx in indices:
        input_data = X_raw[idx:idx+1].astype(np.float32)
        
        # Inference
        probs = sess.run(None, {input_name: input_data})[0][0]
        pred_class = np.argmax(probs)
        
        all_signals.append(X_raw[idx, CHANNEL_TO_PLOT, :, 0])
        all_probs.append(probs)
        all_preds.append(pred_class)
        all_actuals.append(y_raw[idx])

    # 3. Plotting
    print("[3] Generating premium visualization...")
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.2]})
    fig.patch.set_facecolor('#121212')
    
    # --- Top Plot: EEG Signal ---
    flat_signal = np.concatenate(all_signals)
    time_axis = np.linspace(0, NUM_SAMPLES_TO_PLOT, len(flat_signal))
    
    ax1.plot(time_axis, flat_signal, color='#00E5FF', linewidth=0.8, alpha=0.9)
    ax1.set_title(f"Real-time EEG Signal (Channel {CHANNEL_TO_PLOT})", fontsize=14, color='#00E5FF', pad=15)
    ax1.set_ylabel("Amplitude (uV)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xlim(0, NUM_SAMPLES_TO_PLOT)
    
    # เพิ่มแถบสีพื้นหลังตาม Prediction
    for i in range(NUM_SAMPLES_TO_PLOT):
        color = COLORS[all_preds[i]]
        ax1.axvspan(i, i+1, color=color, alpha=0.1)
        # แสดง Text คำทำนายด้านบน
        ax1.text(i + 0.5, np.max(flat_signal) * 1.1, CLASS_LABELS[all_preds[i]], 
                 color=color, fontsize=9, ha='center', fontweight='bold')

    # --- Bottom Plot: Confidence Probability ---
    all_probs = np.array(all_probs)
    x_indices = np.arange(NUM_SAMPLES_TO_PLOT) + 0.5
    
    width = 0.25
    ax2.bar(x_indices - width, all_probs[:, 0], width, label=CLASS_LABELS[0], color=COLORS[0], alpha=0.8)
    ax2.bar(x_indices, all_probs[:, 1], width, label=CLASS_LABELS[1], color=COLORS[1], alpha=0.8)
    ax2.bar(x_indices + width, all_probs[:, 2], width, label=CLASS_LABELS[2], color=COLORS[2], alpha=0.8)
    
    ax2.set_title("Inference Confidence Score", fontsize=14, color='#FFFFFF', pad=15)
    ax2.set_ylabel("Probability", fontsize=10)
    ax2.set_xlabel("Time Segment (Seconds)", fontsize=10)
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(range(NUM_SAMPLES_TO_PLOT + 1))
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', frameon=True, facecolor='#1E1E1E')

    # ตรวจสอบว่าทำนายถูกไหม (ถ้าผิดให้วงกลมจุดนั้น)
    for i in range(NUM_SAMPLES_TO_PLOT):
        if all_preds[i] != all_actuals[i]:
            ax2.annotate('!', xy=(i+0.5, 1.02), color='yellow', fontsize=15, ha='center', fontweight='bold')

    plt.tight_layout()
    output_file = "inference_visualization.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[SUCCESS] Visualization saved to: {output_file}")
    
    # แสดงผลสถิติเล็กน้อย
    accuracy = np.mean(np.array(all_preds) == np.array(all_actuals)) * 100
    print(f"Session Accuracy: {accuracy:.2f}%")
    print("==================================================")

    # พยายามเปิดไฟล์ (สำหรับ Windows)
    try:
        os.startfile(output_file)
    except:
        pass

if __name__ == "__main__":
    visualize_inference()
