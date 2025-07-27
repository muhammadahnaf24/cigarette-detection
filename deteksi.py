import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import os
from datetime import datetime
from PIL import Image
from streamlit_cropper import st_cropper
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Rokok YOLO", 
    page_icon="🚬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header aplikasi
st.title("🚬 Deteksi Rokok dengan YOLOv8")
st.markdown("Aplikasi deteksi objek rokok menggunakan model YOLOv8")

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    try:
        model = YOLO('rokok14.pt')
        model.to('cuda')   
        st.success("Model berhasil dimuat!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None

# Sidebar untuk pengaturan
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    # Pengaturan deteksi
    st.subheader("Parameter Deteksi")
    confidence_threshold = st.slider(
        "Ambang Keyakinan", 
        0.0, 1.0, 0.5, 0.01,
        help="Nilai minimum kepercayaan untuk mendeteksi objek"
    )
    
    iou_threshold = 0.45
    
    # Status sistem
    st.subheader("Status Sistem")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Status Model", "Aktif")
    with col2:
        st.metric("Ambang Keyakinan", f"{confidence_threshold:.2f}")

# Memuat model
model = load_model()
if model is None:
    st.stop()

# Fungsi deteksi gambar
def detect_image(image):
    try:
        # Convert input to proper numpy array format
        if isinstance(image, Image.Image):
            # Convert PIL Image to numpy array (always to RGB)
            image_np = np.array(image.convert('RGB'))
        else:
            # Handle numpy array input
            if len(image.shape) == 2:  # Grayscale
                image_np = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:  # Assume RGB if 3 channels
                image_np = image.copy()  # Create a copy to avoid modifying original
            else:
                raise ValueError("Unsupported image format")

        # Run YOLO detection
        results = model(image_np, iou=iou_threshold, conf=confidence_threshold)

        # Process detection results
        annotated_image = None
        if results:
            result = results[0]
            
            # Get annotated image directly (YOLOv8 already returns RGB)
            annotated_image = result.plot()

        return annotated_image, results

    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return None, None


# Fungsi deteksi kamera
import time

def run_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.warning("Mencari kamera yang tersedia...")
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                st.success(f"Kamera ditemukan pada indeks {i}")
                break
    
    if not cap.isOpened():
        st.error("Tidak dapat mengakses kamera. Pastikan kamera terhubung.")
        return
    
    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    alert_placeholder = st.empty()
    stop_button = st.button("⏹️ Hentikan Deteksi", key="stop_camera")
    
    prev_time = time.time()
    last_detected_time = 0  # Waktu terakhir rokok terdeteksi
    alert_duration = 1      # Detik peringatan tetap muncul

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengambil frame dari kamera")
            break
        
        # Resize frame agar tidak terlalu besar
        height, width = frame.shape[:2]
        new_width = 960
        aspect_ratio = height / width
        new_height = int(new_width * aspect_ratio)
        frame = cv2.resize(frame, (new_width, new_height))

        # Hitung FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        # Deteksi menggunakan YOLOv8
        results = model(frame, iou=iou_threshold, conf=confidence_threshold)
        
        for result in results:
            # Hitung jumlah rokok terdeteksi
            num_cigarettes = 0
            if result.names:
                for cls_id in result.boxes.cls:
                    class_name = result.names[int(cls_id)]
                    if "rokok" in class_name.lower():
                        num_cigarettes += 1

            # Plot hasil deteksi
            annotated_frame = result.plot()
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB")

            # Jika rokok terdeteksi, perbarui waktu deteksi terakhir
            if num_cigarettes > 0:
                last_detected_time = time.time()

            # Logika tampilkan peringatan jika masih dalam durasi alert
            if time.time() - last_detected_time <= alert_duration:
                alert_placeholder.warning(
                    f"🚨 Rokok terdeteksi! (Jumlah: {num_cigarettes})", 
                    icon="⚠️"
                )
            else:
                alert_placeholder.empty()

            # Tampilkan info
            info_placeholder.markdown(
                f"""
                **Jumlah Rokok Terdeteksi:** {num_cigarettes}  
                **FPS (Frame per Second):** {fps:.2f}
                """
            )

    cap.release()



# Fungsi deteksi video
def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    # Placeholder untuk UI
    frame_placeholder = st.empty()
    progress_bar = st.progress(0)
    alert_placeholder = st.empty()
    info_placeholder = st.empty()  # ✅ untuk FPS & jumlah deteksi
    stop_button = st.button("⏹️ Hentikan Pemrosesan", key="stop_video")

    frame_count = 0
    last_detected_time = 0
    alert_duration = 3
    prev_time = time.time()

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        # Hitung FPS real-time
        current_time = time.time()
        fps_processing = 1.0 / (current_time - prev_time)
        prev_time = current_time

        if frame_count % 2 == 0:  # Skip frame untuk percepatan
            results = model(frame, iou=iou_threshold, conf=confidence_threshold)

            for result in results:
                # Hitung jumlah rokok
                num_cigarettes = 0
                if result.names:
                    for cls_id in result.boxes.cls:
                        class_name = result.names[int(cls_id)]
                        if "rokok" in class_name.lower():
                            num_cigarettes += 1

                # Jika terdeteksi, update waktu
                if num_cigarettes > 0:
                    last_detected_time = time.time()

                # Gambar hasil deteksi
                annotated_frame = result.plot()
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB")

                # Peringatan
                if time.time() - last_detected_time <= alert_duration:
                    alert_placeholder.warning(
                        f"🚨 Rokok terdeteksi! (Jumlah: {num_cigarettes})",
                        icon="⚠️"
                    )
                else:
                    alert_placeholder.empty()

                # ✅ Info tambahan (FPS & Jumlah Deteksi)
                info_placeholder.markdown(
                    f"""
                    **Jumlah Rokok Terdeteksi:** {num_cigarettes}  
                    **FPS Proses:** {fps_processing:.2f}  
                    **FPS Video Asli:** {fps_video:.2f}  
                    """
                )

    cap.release()
    os.unlink(tfile.name)
    st.success("Pemrosesan video selesai!")



# Tab utama
tab1, tab2, tab3 = st.tabs(["📷 Kamera Langsung", "🎥 Unggah Video", "🖼️ Analisis Gambar"])

with tab1:
    st.header("Deteksi Kamera Langsung")
    st.markdown("Gunakan kamera perangkat Anda untuk deteksi objek secara real-time.")
    
    camera_options = {
        "Kamera Utama": 0,
        "Kamera Kedua": 1,
        "Kamera Eksternal": 2
    }
    
    selected_camera = st.selectbox("Pilih Kamera", list(camera_options.keys()))

    if st.button("🎥 Mulai Deteksi", key="start_camera"):
        run_camera(camera_options[selected_camera])

with tab2:
    st.header("🎞️ Analisis Video")
    st.markdown("Unggah video untuk dianalisis menggunakan model deteksi YOLOv8.")

    uploaded_video = st.file_uploader(
        "📁 Pilih file video", 
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=False
    )

    if uploaded_video:
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.metric("📄 Nama File", uploaded_video.name)
        with col_info2:
            st.metric("📦 Ukuran", f"{uploaded_video.size / (1024*1024):.1f} MB")

        st.markdown("---")
        st.subheader("🎬 Pratinjau dan Hasil Deteksi Video")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎥 Video Asli**")
            st.video(uploaded_video, format="video/mp4", start_time=0)
            analyze_video = st.button("🚀 Mulai Analisis Video", key="analyze_video")

        with col2:
            st.markdown("**🎥 Video Hasil Deteksi**")
            process_video(uploaded_video)  # ✅ langsung panggil fungsi


with tab3:
    st.header("📷 Analisis Gambar")
    st.markdown("Unggah gambar untuk dianalisis dengan model deteksi rokok.")

    uploaded_image = st.file_uploader(
        "📁 Pilih file gambar", 
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False
    )

    if uploaded_image:
        image = Image.open(uploaded_image)

        # Tampilkan gambar asli
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🖼️ Gambar Asli**")
            st.image(image, caption="Gambar Asli", use_container_width=True)

            analyze = st.button("Mulai Analisis", key="analyze_image")
        with col2:
            if analyze:
                with st.spinner("Menganalisis gambar..."):
       
                    annotated_image, results = detect_image(image)

                    if annotated_image is not None:
                        st.markdown("**🖼️ Gambar Hasil Deteksi**")
                        st.image(annotated_image, caption="Hasil Deteksi", width=640)

                        detection_data = []
                        for result in results:
                            for box in result.boxes:
                                cls_id = int(box.cls.item())
                                conf = box.conf.item()
                                cls_name = result.names[cls_id]
                                detection_data.append((cls_name, conf))

                        if detection_data:
                            st.subheader("📊 Statistik Deteksi")
                            detection_summary = {}
                            for obj, conf in detection_data:
                                detection_summary.setdefault(obj, []).append(conf)

                            for obj, confs in detection_summary.items():
                                avg_conf = sum(confs) / len(confs)
                                st.success(f"{obj}: {len(confs)} deteksi (Rata-rata keyakinan: {avg_conf:.1%})")
                        else:
                            st.warning("🚫 Tidak ada objek yang terdeteksi.")

            


# Footer
st.markdown("---")
st.caption(f"© {datetime.now().year} Deteksi Rokok YOLO | Dibangun dengan Streamlit dan YOLOv8")