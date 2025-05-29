import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
import os
from datetime import datetime
from PIL import Image # Diperlukan untuk streamlit-cropper
from streamlit_cropper import st_cropper # Import komponen cropper

# Set page config with modern styling
st.set_page_config(
    page_title="YOLO Cigarette Detection", 
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Dark Theme CSS
st.markdown("""
    <style>
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styling */
        .stApp {
            background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
            font-family: 'Inter', sans-serif;
            color: #e6edf3;
        }
        
        /* Hide default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Custom header */
        .custom-header {
            background: linear-gradient(90deg, #1f6feb 0%, #8b949e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2rem;
            letter-spacing: -0.02em;
        }
        
        .custom-subtitle {
            text-align: center;
            color: #8b949e;
            font-size: 1.1rem;
            margin-bottom: 3rem;
            font-weight: 400;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: #161b22;
            border-right: 1px solid #30363d;
        }
        
        .sidebar-header {
            background: linear-gradient(135deg, #1f6feb, #8b949e);
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            text-align: center;
            color: white;
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(35, 134, 54, 0.3);
            width: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #2ea043 0%, #238636 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(35, 134, 54, 0.4);
        }
        
        .stop-button > button {
            background: linear-gradient(135deg, #da3633 0%, #f85149 100%) !important;
            box-shadow: 0 4px 12px rgba(218, 54, 51, 0.3) !important;
        }
        
        .stop-button > button:hover {
            background: linear-gradient(135deg, #f85149 0%, #da3633 100%) !important;
            box-shadow: 0 6px 20px rgba(218, 54, 51, 0.4) !important;
        }
        
        /* Sliders */
        .stSlider > div > div > div > div {
            background: #1f6feb;
        }
        
        .stSlider > div > div > div > div > div {
            background: #1f6feb;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 8px;
        }
        
        /* File uploader */
        .stFileUploader > div {
            background: #21262d;
            border: 2px dashed #30363d;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }
        
        .stFileUploader > div:hover {
            border-color: #1f6feb;
            background: #0d1117;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: #21262d;
            border-radius: 12px;
            padding: 0.5rem;
            margin-bottom: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: #8b949e;
            font-weight: 500;
            padding: 0.75rem 1.5rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1f6feb, #8b949e);
            color: white;
        }
        
        /* Cards */
        .detection-card {
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #21262d 0%, #161b22 100%);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            margin: 0.5rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1f6feb;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #8b949e;
            margin-top: 0.5rem;
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active {
            background: #2ea043;
            box-shadow: 0 0 8px rgba(46, 160, 67, 0.6);
        }
        
        .status-inactive {
            background: #8b949e;
        }
        
        /* Image containers */
        .image-container {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            border: 1px solid #30363d;
        }
        
        /* Detection results */
        .detection-result {
            background: #0d1117;
            border-left: 4px solid #1f6feb;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
        }
        
        .detection-object {
            color: #e6edf3;
            font-weight: 600;
        }
        
        .detection-confidence {
            color: #2ea043;
            font-weight: 500;
        }
        
        /* Animations */
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 2s infinite;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .custom-header {
                font-size: 2rem;
            }
            
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Custom header
st.markdown('<h1 class="custom-header">YOLO Cigarette Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="custom-subtitle">Advanced Object Detection with Professional Interface</p>', unsafe_allow_html=True)

# Sidebar with professional styling
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration Panel</div>', unsafe_allow_html=True)
    
    st.markdown("### Detection Parameters")
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.5, 0.01,
        help="Minimum confidence score for object detection"
    )
    
    iou_threshold = st.slider(
        "IoU Threshold", 
        0.0, 1.0, 0.45, 0.01,
        help="Intersection over Union threshold for non-maximum suppression"
    )
    
    st.markdown("### Camera Configuration")
    camera_options = {
        "📷 Default Camera": 0,
        "🎥 OBS Virtual Camera": 1,
        "📹 External Camera": 2
    }
    
    selected_camera = st.selectbox(
        "Camera Source", 
        list(camera_options.keys()),
        help="Select your preferred camera input"
    )
    
    # Model status
    st.markdown("### System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
            <div class="metric-card">
                <div class="metric-value">✅</div>
                <div class="metric-label">Model Ready</div>
            </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{confidence_threshold:.2f}</div>
                <div class="metric-label">Confidence</div>
            </div>
        ''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load YOLO model with caching"""
    try:
        model = YOLO('best3.pt')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Load model
with st.spinner("🔄 Loading AI model..."):
    model = load_model()

if model is None:
    st.error("❌ Model loading failed. Please check your model file.")
    st.stop()

def detect_image(image):
    """Process image detection with error handling"""
    try:
        # Image preprocessing
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize for consistent processing
        resized_image = cv2.resize(image, (1280, 720))
        
        # Run detection
        results = model(resized_image, iou=iou_threshold, conf=confidence_threshold)
        
        # Get annotated image
        for result in results:
            annotated_image = result.plot()
        
        return annotated_image, results
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return None, None

def video_capture(camera_idx=0):
    """Handle video capture with professional UI"""
    cap = cv2.VideoCapture(camera_idx)
    
    if not cap.isOpened():
        st.warning(f"🔍 Searching for available cameras...")
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                st.success(f"✅ Camera found at index {i}")
                break

    if not cap.isOpened():
        st.error("❌ No cameras detected. Please check your camera connections.")
        return

    # UI elements
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        frame_placeholder = st.empty()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        stop_button = st.button("🛑 Stop Detection", key="stop_camera", help="Click to stop camera detection")

    # Detection loop
    frame_count = 0
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture video frame")
            break

        frame_count += 1
        
        # Process every few frames for performance
        if frame_count % 2 == 0:
            resized_frame = cv2.resize(frame, (1280, 720))
            results = model(resized_frame, iou=iou_threshold, conf=confidence_threshold)
            
            for result in results:
                annotated_frame = result.plot()
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                with frame_placeholder.container():
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(rgb_frame, channels="RGB", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    cap.release()
    cv2.destroyAllWindows()

# Main content with professional tabs
tab1, tab2, tab3 = st.tabs(["📷 Live Camera", "🎞️ Video Upload", "🖼️ Image Analysis"])

with tab1:
    st.markdown('<div class="detection-card">', unsafe_allow_html=True)
    st.markdown("### 📷 Real-Time Detection")
    st.markdown("Start live camera detection with real-time object recognition")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶️ Start Live Detection", key="start_camera", help="Begin real-time object detection"):
            camera_idx = camera_options[selected_camera]
            video_capture(camera_idx)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="detection-card">', unsafe_allow_html=True)
    st.markdown("### 🎞️ Video Analysis")
    st.markdown("Upload and analyze video files for object detection")
    
    uploaded_video = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov", "mkv"], 
        key="video_uploader",
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_video is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**File:** {uploaded_video.name}")
            st.markdown(f"**Size:** {uploaded_video.size / 1024 / 1024:.1f} MB")
        
        with col2:
            process_button = st.button("🚀 Process Video", key="process_video")
        
        if process_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                cap = cv2.VideoCapture(tfile.name)
                
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                # Get total frames
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0
                
                stop_button = st.button("🛑 Stop Processing", key="stop_video")

                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    
                    if frame_count % 3 == 0:  # Process every 3rd frame
                        resized_frame = cv2.resize(frame, (1280, 720))
                        results = model(resized_frame, iou=iou_threshold, conf=confidence_threshold)
                        
                        for result in results:
                            annotated_frame = result.plot()
                            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            
                            with frame_placeholder.container():
                                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                                st.image(rgb_frame, channels="RGB", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                cap.release()
                os.unlink(tfile.name)
                st.success("✅ Video processing completed!")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="detection-card">', unsafe_allow_html=True)
    st.markdown("### 🖼️ Image Detection")
    st.markdown("Upload images for detailed object analysis")
    
    uploaded_image = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png", "bmp", "webp"], 
        key="image_uploader",
        help="Supported formats: JPG, PNG, BMP, WebP"
    )
    
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 Original Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if st.button("🔍 Analyze Image", key="analyze_image", help="Start object detection analysis"):
                with st.spinner("🤖 AI is analyzing your image..."):
                    annotated_image, results = detect_image(image)
                    
                    if annotated_image is not None:
                        st.markdown("#### 🎯 Detection Results")
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(annotated_image, channels="RGB", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detection results summary
        if uploaded_image is not None:
            st.markdown("#### 📊 Detection Summary")
            
            # Run detection for summary
            results = model(image, iou=iou_threshold, conf=confidence_threshold)
            
            for result in results:
                if len(result.boxes) > 0:
                    detection_data = []
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        conf = box.conf.item()
                        cls_name = result.names[cls_id]
                        detection_data.append((cls_name, conf))
                    
                    # Group detections
                    detection_summary = {}
                    for obj, conf in detection_data:
                        if obj not in detection_summary:
                            detection_summary[obj] = []
                        detection_summary[obj].append(conf)
                    
                    # Display results
                    for obj, confidences in detection_summary.items():
                        avg_conf = sum(confidences) / len(confidences)
                        count = len(confidences)
                        
                        st.markdown(f'''
                            <div class="detection-result">
                                <span class="detection-object">{obj}</span> 
                                <span style="color: #8b949e;">×{count}</span> - 
                                <span class="detection-confidence">{avg_conf:.1%} confidence</span>
                            </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.markdown('''
                        <div class="detection-result">
                            <span style="color: #8b949e;">No objects detected</span>
                        </div>
                    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown(f'''
        <div style="text-align: center; color: #8b949e; font-size: 0.9rem;">
            Cigarette Detection | Powered by YOLOv8<br>
            <small>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</small>
        </div>
    ''', unsafe_allow_html=True)