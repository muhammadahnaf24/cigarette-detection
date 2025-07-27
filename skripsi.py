import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Pose Tangan Tari Tradisional",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul aplikasi
st.title("🖐️ Deteksi Pose Tangan Tari Tradisional")
st.markdown("Aplikasi deteksi pose tangan untuk tari tradisional menggunakan YOLOv8")

# Sidebar untuk pengaturan
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    confidence_threshold = st.slider("Ambang Keyakinan", 0.0, 1.0, 0.5, 0.01)
    
    st.header("📁 Mode Input")
    input_type = st.radio("Pilih sumber input:", 
                         ["Gambar", "Webcam"])

# Load model (ganti dengan model pose YOLOv8 yang sudah dilatih)
@st.cache_resource
def load_model():
    try:
        model = YOLO('ihsan.pt')  # Model pose detection bawaan YOLOv8
        st.sidebar.success("Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

model = load_model()

# Fungsi untuk deteksi pose
def detect_pose(image):
    results = model(image, conf=confidence_threshold)
    annotated_image = results[0].plot() if results else image
    return annotated_image, results

# Mode Input Gambar
if input_type == "Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar tangan tari tradisional", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Gambar Input")
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Asli", use_column_width=True)
            
        with col2:
            st.header("Hasil Deteksi")
            with st.spinner("Mendeteksi pose..."):
                # Convert PIL Image to OpenCV format
                image_cv = np.array(image.convert('RGB'))
                annotated_image, results = detect_pose(image_cv)
                st.image(annotated_image, caption="Pose Terdeteksi", use_column_width=True)
                
                # Tampilkan keypoints jika terdeteksi
                if results and results[0].keypoints is not None:
                    st.subheader("Keypoints Tangan")
                    keypoints = results[0].keypoints.xy[0].cpu().numpy()
                    st.write(f"Jumlah keypoints terdeteksi: {len(keypoints)}")
                    st.write(keypoints)

# Mode Input Webcam
elif input_type == "Webcam":
    st.header("Deteksi Pose Real-time")
    run = st.checkbox("Aktifkan Kamera")
    
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(1)  # Kamera default
    
    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengambil frame dari kamera")
            break
            
        # Deteksi pose
        results = model(frame, conf=confidence_threshold)
        annotated_frame = results[0].plot() if results else frame
        
        # Convert BGR to RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(annotated_frame)
        
    cap.release()
    if not run:
        st.warning("Kamera nonaktif")

# Catatan kaki
st.markdown("---")
st.caption("Aplikasi deteksi pose tangan tari tradisional | Dibangun dengan YOLOv8 dan Streamlit")