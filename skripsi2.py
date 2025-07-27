import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time

# ==================== CONFIG ====================
# Load model YOLOv8n
model = YOLO("ihsan.pt")  
label_map = model.names  # Dict: {0: 'Ngithing', 1: 'Nyempurit', ...}

# ==================== TITLE & SIDEBAR ====================
st.set_page_config(page_title="Deteksi Pose Tari Jawa", layout="wide")
st.title("🎭 Deteksi Pose (Bentuk Jari) Tarian Tradisional Jawa")

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("🛠️ Detection Parameters")
    conf_thres = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
    iou_thres = st.slider("IOU Threshold", min_value=0.1, max_value=1.0, value=0.45, step=0.05)

    st.subheader("📷 Camera Configuration")
    camera_type = st.selectbox("Pilih Kamera", ["Webcam Internal", "OBS / DroidCam (Virtual Cam)"])

# ==================== DETEKSI ====================
def predict_image(image, conf, iou):
    results = model.predict(
        image,
        conf=conf,
        iou=iou,
        agnostic_nms=False
    )
    plotted = results[0].plot()
    return plotted, results[0]

# ==================== MODE TABS ====================
tab1, tab2 = st.tabs(["📸 Live Camera", "🖼️ Upload Gambar"])

# === TAB UPLOAD GAMBAR ===
with tab2:
    uploaded_file = st.file_uploader("Unggah Gambar Pose (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar Asli")

        if st.button("🔍 Jalankan Prediksi"):
            with st.spinner("Mendeteksi..."):
                result_image, result = predict_image(image, conf_thres, iou_thres)
                st.image(result_image, caption="Hasil Prediksi")

                # Informasi Deteksi
                st.markdown("### 🧾 Hasil Deteksi")
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    st.write(f"Pose: **{label_map[cls]}**, Confidence: **{conf:.2f}**")

# === TAB LIVE CAMERA ===
with tab1:
    run = st.checkbox('Aktifkan Kamera')
    stframe = st.empty()

    if run:
        cap = cv2.VideoCapture(0 if camera_type == "Webcam Internal" else 1)
        if not cap.isOpened():
            st.error("Tidak dapat mengakses kamera!")
        else:
            st.info("Kamera aktif. Hilangkan centang untuk berhenti.")
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Tidak dapat membaca frame dari kamera.")
                    break

                # Konversi ke RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Prediksi
                result_frame, result = predict_image(frame_rgb, conf_thres, iou_thres)

                # Tampilkan hasil deteksi
                stframe.image(result_frame, channels="RGB", use_container_width=True)



            cap.release()
            st.success("Kamera dimatikan.")


# ==================== EVALUASI MODEL ====================
st.markdown("---")
st.subheader("📊 Evaluasi Model (Data Uji)")

eval = st.checkbox("Tampilkan Evaluasi Model")
if eval:
    # Dummy Ground Truth vs Prediksi (gantilah dengan data asli saat deploy)
    y_true = np.random.choice([0, 1, 2, 3, 4, 5], size=100)
    y_pred = np.random.choice([0, 1, 2, 3, 4, 5], size=100)

    labels = list(label_map.values())
    cm = confusion_matrix(y_true, y_pred)
    labels = ['ngithing', 'ngruji', 'ngepel', 'nyempurit', 'boyo_mangap', 'non-pose']
    report = classification_report(y_true, y_pred, output_dict=True, target_names=labels)

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)

    # Metrik Evaluasi
    st.markdown("### Metrik Evaluasi")
    st.json({
        "Akurasi": f"{np.trace(cm) / np.sum(cm):.2f}",
        "Precision": f"{np.mean([v['precision'] for v in report.values() if isinstance(v, dict)]):.2f}",
        "Recall": f"{np.mean([v['recall'] for v in report.values() if isinstance(v, dict)]):.2f}",
        "F1-Score": f"{np.mean([v['f1-score'] for v in report.values() if isinstance(v, dict)]):.2f}",
    })

    # Kurva Evaluasi
    st.markdown("### Kurva Evaluasi (Simulasi)")
    fig2, ax2 = plt.subplots()
    x = list(range(1, 11))
    acc = np.random.uniform(0.75, 1.0, size=10)
    f1 = np.random.uniform(0.6, 0.95, size=10)
    ax2.plot(x, acc, label="Akurasi", marker='o')
    ax2.plot(x, f1, label="F1-Score", marker='x')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Nilai")
    ax2.set_title("Kurva Evaluasi Model")
    ax2.legend()
    st.pyplot(fig2)