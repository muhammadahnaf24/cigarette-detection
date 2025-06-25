import cv2
import numpy as np

# Baca gambar (ubah 'gambar_input.jpg' ke path file-mu)
img = cv2.imread('kernel.png')

# Ubah ke grayscale (jika ingin nilai intensitas tunggal seperti pada Gambar 3.7)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tentukan area yang ingin diambil (misalnya koordinat kiri atas (x,y) dan ukuran 5x5)
x, y = 100, 100  # ganti sesuai kebutuhan
w, h = 5, 5

# Ambil nilai piksel dari area tersebut
region = gray[y:y+h, x:x+w]

# Tampilkan hasilnya
print("Nilai piksel (grayscale) dari area yang dipilih:")
print(region)
