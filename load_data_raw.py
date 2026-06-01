# สคริปต์โหลดข้อมูลสัญญาณดิบ (Raw Signal) สำหรับ EEGNet
import mne
import numpy as np
import os
import scipy.signal

# แก้ไข Path ให้ตรงกับความจริง
DATA_PATH = "C:/9 ML Pipeline"

# ---------------------------------------------------------
# การเลือกกลุ่มตัวอย่างแบบ AUTO_QUALITY (คัดคนคุณภาพดีสุด 25 คน)
# ---------------------------------------------------------
import json
import random
try:
    with open("top_80_subjects.json", "r") as f:
        top_80 = json.load(f)
    
    NUM_SUBJECTS_TO_SAMPLE = 50
    subjects = random.sample(top_80, min(NUM_SUBJECTS_TO_SAMPLE, len(top_80)))
    subjects.sort()
    print(f"[INFO] AUTO_QUALITY MODE: Randomly sampled {len(subjects)} from Top 80 best subjects.")
except FileNotFoundError:
    print("[ERROR] ไม่พบไฟล์ 'top_80_subjects.json'")
    subjects = []

X_raw = []
y_raw = []

for subject in subjects:
    subject_path = os.path.join(DATA_PATH, subject)
    if not os.path.exists(subject_path):
        # ลองหาแบบไม่มีโฟลเดอร์ย่อย (เผื่อโครงสร้างไฟล์ต่างกัน)
        continue

    subject_segments = []
    subject_labels = []
    target_runs = ["R03", "R04", "R07", "R08", "R11", "R12"]
    
    for file in os.listdir(subject_path):
        if file.endswith(".edf") and any(r in file for r in target_runs):
            file_path = os.path.join(subject_path, file)
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                # Band-pass Filter (มาตรฐาน BCI)
                raw.filter(l_freq=4.0, h_freq=40.0, fir_design='firwin', verbose=False)
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                data = raw.get_data()[:64, :] # 64 Channels
                
                for i in range(len(events)):
                    start_samp = events[i][0]
                    code = events[i][2]
                    event_name = [k for k, v in event_id.items() if v == code][0]
                    
                    if event_name == 'T1': label = 0 # Left
                    elif event_name == 'T2': label = 1 # Right
                    elif event_name == 'T0': label = 2 # Rest
                    else: continue

                    window_size = 256 # 1.6 วินาที
                    if start_samp + window_size < data.shape[1]:
                        segment = data[:, start_samp : start_samp + window_size]
                        subject_segments.append(segment)
                        subject_labels.append(label)
                        
                        # Data Augmentation (Oversampling)
                        if label in [0, 1]:
                            if start_samp + window_size + 32 < data.shape[1]:
                                subject_segments.append(data[:, start_samp+32 : start_samp+window_size+32])
                                subject_labels.append(label)
            except:
                continue

    # Per-Subject Normalization
    if len(subject_segments) > 0:
        subj_data = np.array(subject_segments)
        mean = np.mean(subj_data, axis=(0, 2), keepdims=True)
        std = np.std(subj_data, axis=(0, 2), keepdims=True) + 1e-6
        subj_data_norm = (subj_data - mean) / std
        
        for i in range(len(subj_data_norm)):
            X_raw.append(subj_data_norm[i])
            y_raw.append(subject_labels[i])

X_raw = np.array(X_raw, dtype=np.float32)
# ปรับ Shape ให้เข้ากับ EEGNet: (Samples, Channels, TimePoints, 1)
X_raw = X_raw.reshape(X_raw.shape[0], 64, 256, 1)
y_raw = np.array(y_raw, dtype=np.int32)

print(f"Final Raw Data Shape: {X_raw.shape}")
print(f"Final Labels Shape: {y_raw.shape}")

# บันทึกข้อมูลเพื่อนำไปเทรน
np.save('X_raw.npy', X_raw)
np.save('y_raw.npy', y_raw)
print("[SUCCESS] Raw Data saved as X_raw.npy and y_raw.npy")
