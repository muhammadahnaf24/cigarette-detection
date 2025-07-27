import cv2
import albumentations as A
import numpy as np
import os
from glob import glob

# Path folder gambar
source_folder = "dataset/original"  # ganti dengan folder gambar kamu
output_folder = "dataset/augmentasi"
os.makedirs(output_folder, exist_ok=True)

# Fungsi untuk Mosaic
# Fungsi Mosaic yang resize gambar dulu
def mosaic(images, size=(640, 640)):
    resized = [cv2.resize(img, size) for img in images]
    top = np.concatenate(resized[:2], axis=1)
    bottom = np.concatenate(resized[2:], axis=1)
    return np.concatenate([top, bottom], axis=0)


# Dapatkan semua file gambar
image_paths = glob(os.path.join(source_folder, "*.jpg"))  # bisa juga "*.png"

# Loop semua gambar
for path in image_paths:
    filename = os.path.splitext(os.path.basename(path))[0]
    image = cv2.imread(path)

    # ROTASI ±15°
    rotated = A.Affine(rotate=(-15, 15), mode=cv2.BORDER_CONSTANT, cval=0)(image=image)['image']
    cv2.imwrite(f"{output_folder}/{filename}_rotated.jpg", rotated)

    # SCALE (zoom in 1.3x atau zoom out 0.7x)
    scaled = A.Affine(scale=(1.3, 1.3), mode=cv2.BORDER_CONSTANT, cval=0)(image=image)['image']
    cv2.imwrite(f"{output_folder}/{filename}_scaled.jpg", scaled)


    # 2. HUE SHIFT -25
    hue_img = A.HueSaturationValue(hue_shift_limit=(-25, -25), sat_shift_limit=0, val_shift_limit=0, p=1.0)(image=image)['image']
    cv2.imwrite(f"{output_folder}/{filename}_hue.jpg", hue_img)

    # 3. SATURATION +30
    sat_img = A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(30, 30), val_shift_limit=0, p=1.0)(image=image)['image']
    cv2.imwrite(f"{output_folder}/{filename}_saturation.jpg", sat_img)

    # Resize back to original size (karena RandomScale bisa mengubah ukuran)
    scaled = cv2.resize(scaled, (image.shape[1], image.shape[0]))
    cv2.imwrite(f"{output_folder}/{filename}_scaled.jpg", scaled)

# 5. MOSAIC: per 4 gambar, jika tersedia
for i in range(0, len(image_paths), 4):
    imgs = []
    for p in image_paths[i:i+4]:
        img = cv2.imread(p)
        if img is not None:
            imgs.append(img)
    if len(imgs) == 4:
        mosaic_img = mosaic(imgs)
        cv2.imwrite(f"{output_folder}/mosaic_{i//4}.jpg", mosaic_img)

print("✅ Semua augmentasi selesai dan disimpan ke folder 'augmented'")
