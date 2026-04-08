import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. INPUT ---
image_path = r'C:\Users\devar\Desktop\IP_assignment\Images\Screenshot 2025-09-06 004327.png'
print(f"Processing image: {image_path}")
I_RGB = cv2.imread(image_path)

if I_RGB is None:
    raise FileNotFoundError(f"Could not read image file: {image_path}")

# --- 2. PREPROCESSING ---
# Convert to HSV and equalize Value channel
I_HSV = cv2.cvtColor(I_RGB, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(I_HSV)
v_eq = cv2.equalizeHist(v)
I_HSV_eq = cv2.merge([h, s, v_eq])
I_ContrastEnhanced = cv2.cvtColor(I_HSV_eq, cv2.COLOR_HSV2BGR)

# Convert to grayscale
I_Gray = cv2.cvtColor(I_ContrastEnhanced, cv2.COLOR_BGR2GRAY)

# Contrast stretch (equivalent to imadjust)
p2, p98 = np.percentile(I_Gray, (1, 99))
I_Adjusted = cv2.normalize(I_Gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Median filter for noise removal
I_Median = cv2.medianBlur(I_Adjusted, 3)

# Bilateral filter for smoother edges without losing detail
I_Smooth = cv2.bilateralFilter(I_Median, 9, 75, 75)

# --- 3. EDGE DETECTION (Canny + Morph cleanup) ---
edges = cv2.Canny(I_Smooth, 50, 150)  # tune thresholds for edge density

# Morphological closing to fill small gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
edges_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
edges_clean = cv2.morphologyEx(edges_clean, cv2.MORPH_OPEN, kernel)

# --- 4. CIRCLE DETECTION (Hough Transform) ---
Rmin, Rmax = 15, 120
circles = cv2.HoughCircles(edges_clean, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                           param1=100, param2=30,
                           minRadius=Rmin, maxRadius=Rmax)

num_tanks = 0
detected = []

if circles is not None:
    circles = np.uint16(np.around(circles[0, :]))
    num_tanks = len(circles)
    print(f"Detected {num_tanks} possible tanks.")
else:
    print("No tanks detected.")
    circles = []

# --- 5. VISUALIZATION ---
vis_img = I_RGB.copy()

for i, circle in enumerate(circles, start=1):
    x, y, r = circle
    cv2.circle(vis_img, (x, y), r, (255, 0, 0), 2)
    cv2.circle(vis_img, (x, y), 2, (0, 0, 255), 3)
    cv2.putText(vis_img, str(i), (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)
    detected.append((x, y, r))

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
plt.title(f"Detected {num_tanks} Tanks")
plt.axis('off')
plt.show()

# --- 6. SHADOW PERCENTAGE CALCULATION ---
if len(detected) > 0:
    print("\n--- Shadow Analysis ---")
    I_HSV_orig = cv2.cvtColor(I_RGB, cv2.COLOR_BGR2HSV)
    V = I_HSV_orig[:, :, 2].astype(np.float32) / 255.0

    shadow_threshold = 0.5  # Brightness threshold for shadow
    R, C = V.shape

    for idx, (x, y, r) in enumerate(detected, start=1):
        Y, X = np.ogrid[:R, :C]
        mask = (X - x)**2 + (Y - y)**2 <= r**2
        vals = V[mask]
        if len(vals) == 0:
            continue
        shadow_pixels = np.sum(vals < shadow_threshold)
        shadow_percent = (shadow_pixels / len(vals)) * 100
        print(f"Tank {idx} (R={r}): {shadow_percent:.2f}% Shadow")

# --- 7. Edge Map Display ---
plt.figure(figsize=(8, 8))
plt.imshow(edges_clean, cmap='gray')
plt.title("Edge Map Used for Detection")
plt.axis('off')
plt.show()
