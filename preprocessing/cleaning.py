# Ponowne załadowanie bibliotek po resecie
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Ścieżka do przykładowego obrazu cytologicznego
image_path = r"C:\Users\aleks\OneDrive\Documents\inzynierka\data\LBC_slides\LSIL\pow. 10\34a.bmp"
if not os.path.exists(image_path):
    raise FileNotFoundError("Brak obrazu 'sample_cytology.png'. Prześlij obraz komórki lub slajdu.")

# Wczytaj oryginalny obraz
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Skala szarości
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Filtr Gaussa
gauss = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. Median filtering
median = cv2.medianBlur(gauss, 5)

# 4. Bilateral filtering (kolorowy, zachowuje krawędzie)
bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)

# Wyświetlenie obrazów
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img_rgb)
axs[0].set_title("Oryginał")
axs[0].axis('off')

axs[1].imshow(median, cmap='gray')
axs[1].set_title("Median Blur (szarość)")
axs[1].axis('off')

axs[2].imshow(bilateral_rgb)
axs[2].set_title("Bilateral Filter (kolor)")
axs[2].axis('off')

plt.tight_layout()
plt.show()
