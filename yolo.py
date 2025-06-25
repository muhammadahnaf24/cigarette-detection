import torch
from ultralytics import YOLO

# Muat model
model = YOLO("rokok13.pt")

# Aktifkan mode training
model.train(data="coco8.yaml", epochs=1, imgsz=320)  # Forward/backward pass

# Sekarang gradiens akan terisi
model.info(verbose=True)