import tensorflow as tf
from tensorflow.keras import layers, models, constraints
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import time
import psutil

# ==========================================
# PROJECT INFO
# ==========================================
PROJECT_NAME = "การพัฒนาเกมฟื้นฟูสมรรถภาพทางสมองโดยใช้ Brain-Computer Interface (BCI) สำหรับผู้ป่วยโรคหลอดเลือดสมอง"

print(f"{PROJECT_NAME} Pipeline")



# ==========================================
# HYBRID ARCHITECTURE: LARGE EEGNet + LSTM (~1.5M Params)
# ==========================================
def create_hybrid_eegnet_lstm(nb_classes=3, Chans=64, Samples=256, dropoutRate=0.5):
    input1 = layers.Input(shape=(Chans, Samples, 1))

    # --- 1. EEGNet Block (ขยายฟิลเตอร์ให้ใหญ่ขึ้น) ---
    block1 = layers.Conv2D(32, (1, 64), padding='same', use_bias=False)(input1) # เพิ่มเป็น 32
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.DepthwiseConv2D((Chans, 1), use_bias=False, 
                                   depth_multiplier=4, # เพิ่มเป็น 4
                                   depthwise_constraint=constraints.max_norm(1.))(block1)
    block1 = layers.BatchNormalization()(block1)
    block1 = layers.Activation('elu')(block1)
    block1 = layers.AveragePooling2D((1, 4))(block1)
    block1 = layers.Dropout(dropoutRate)(block1)

    block2 = layers.SeparableConv2D(64, (1, 16), use_bias=False, padding='same')(block1) # เพิ่มเป็น 64
    block2 = layers.BatchNormalization()(block2)
    block2 = layers.Activation('elu')(block2)
    block2 = layers.AveragePooling2D((1, 8))(block2)
    block2 = layers.Dropout(dropoutRate)(block2)

    # Output ของ Block2 จะมีรูปร่าง (None, 1, 8, 64)
    
    # --- 2. LSTM Block (ขยายเซลล์สมองให้แตะ 1.5 ล้าน Params) ---
    reshape = layers.Reshape((8, 64))(block2)
    
    # อัด LSTM ขนาดใหญ่มาก (512 units) = ~1.18 ล้านพารามิเตอร์
    lstm = layers.LSTM(512, return_sequences=False, dropout=0.5, recurrent_dropout=0.3)(reshape)

    # --- 3. Dense Head (ชั้นกลั่นกรองก่อนตอบ) ---
    dense1 = layers.Dense(512, activation='relu', kernel_constraint=constraints.max_norm(0.5))(lstm)
    dense1 = layers.Dropout(0.5)(dense1)
    
    dense2 = layers.Dense(128, activation='relu', kernel_constraint=constraints.max_norm(0.5))(dense1)
    dense2 = layers.Dropout(0.5)(dense2)

    dense_out = layers.Dense(nb_classes, kernel_constraint=constraints.max_norm(0.25))(dense2)
    softmax = layers.Activation('softmax')(dense_out)

    return models.Model(inputs=input1, outputs=softmax)

print("\n" + "="*50)
print("  HYBRID EEGNet-LSTM MODEL ARCHITECTURE")
print("="*50)
model = create_hybrid_eegnet_lstm()

def check_model_vram(model):
    total_params = model.count_params()
    fp32_size_mb = (total_params * 4) / (1024 * 1024)
    train_vram_est = fp32_size_mb * 3
    
    print(f"\n--- Model Resource Analysis ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Weights (FP32): {fp32_size_mb:.2f} MB")
    print(f"Estimated VRAM for Training: {train_vram_est:.2f} MB")

check_model_vram(model)
model.summary()
print("="*50 + "\n")

# --- 2. LOAD DATA ---
print("[INFO] Loading Raw EEG Data...")
if not os.path.exists('X_raw.npy'):
    print("[ERROR] X_raw.npy not found! Please run load_data_raw.py first.")
    exit()

X = np.load('X_raw.npy')
y = np.load('y_raw.npy')
y_cat = to_categorical(y, 3)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.15, random_state=42)

# --- 3. TRAINING ---
# กำหนด Optimizer (ตัวปรับจูนน้ำหนัก) และ Loss Function
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# คลาสพิเศษสำหรับแสดงสถานะ Hardware (CPU/RAM) ขณะเทรนในแต่ละ Epoch
class HardwareMonitorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, 'numpy'): lr = lr.numpy()
        
        cpu_usage = psutil.cpu_percent()
        ram_mb = psutil.virtual_memory().used / (1024 * 1024)
        print(f"\n[Epoch {epoch+1}] LR: {lr:.6f} | CPU: {cpu_usage}% | RAM: {ram_mb:.0f} MB")

print("[INFO] Starting Hybrid EEGNet-LSTM Training...")
t0 = time.time()

# ระบบอัตโนมัติที่จะควบคุมการเทรนในแต่ละ Epoch
callbacks = [
    # 1. EarlyStopping: ถ้า Validation Loss ไม่ดีขึ้นติดต่อกัน 10 ครั้ง จะหยุดเทรนทันทีเพื่อกัน Overfitting
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    
    # 2. ReduceLROnPlateau: ถ้าโมเดลเริ่มเรียนรู้ช้าลง (Loss ไม่ลด) จะลด Learning Rate ลงครึ่งหนึ่ง (factor=0.5)
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    
    # 3. HardwareMonitor: แสดงการใช้ทรัพยากรเครื่อง
    HardwareMonitorCallback()
]

# เริ่มกระบวนการเทรน (The Learning Loop)
history = model.fit(
    X_train, y_train,          # ข้อมูลที่ใช้สอนโมเดล
    epochs=100,                # จำนวนรอบที่จะให้โมเดลอ่านข้อมูลทั้งหมด (สูงสุด 100 รอบ)
    batch_size=64,             # แบ่งข้อมูลเข้าเทรนครั้งละ 64 ตัวอย่าง เพื่อความเร็วและเสถียร
    validation_data=(X_test, y_test), # ข้อมูลชุดทดสอบที่โมเดลไม่เคยเห็น เพื่อวัดความแม่นยำจริง
    callbacks=callbacks        # เรียกใช้งานระบบควบคุมอัตโนมัติที่ตั้งไว้ด้านบน
)

# --- 4. EVALUATION & PLOT ---
print("\n[INFO] Generating Training Graphs...")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], linestyle='--', label='Val Acc')
plt.title('Hybrid Model Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], linestyle='--', label='Val Loss')
plt.title('Hybrid Model Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("hybrid_training_graph.png")

print("\n" + "="*50)
print("       FINAL CLASSIFICATION REPORT")
print("="*50)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, y_pred_classes, target_names=['Left', 'Right', 'Rest']))

cm = confusion_matrix(y_true_classes, y_pred_classes)
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Left', 'Right', 'Rest'], yticklabels=['Left', 'Right', 'Rest'])
plt.title('Hybrid EEGNet-LSTM Confusion Matrix')
plt.savefig('hybrid_confusion_matrix.png')

model.save('bci_hybrid_model.h5')
print("\n[SUCCESS] Hybrid Model and Results saved.")
