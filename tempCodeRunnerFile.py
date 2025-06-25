from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO('rokok13.pt')

# Print model summary seperti gambar
model.info(detailed=True, verbose=True)