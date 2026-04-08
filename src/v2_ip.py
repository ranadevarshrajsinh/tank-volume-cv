import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog

# === Circle detection parameters (unchanged) ===
HOUGH_DP = 1.0
HOUGH_MIN_DIST = 75
HOUGH_PARAM1 = 195
HOUGH_PARAM2 = 45
HOUGH_MIN_RADIUS = 20
HOUGH_MAX_RADIUS = 120

# --- Folder selection ---
root = tk.Tk()
root.withdraw()
print("Select input folder with images:")
input_folder = filedialog.askdirectory(title="Select INPUT folder (images)")

if not input_folder:
    print("No input folder selected — exiting.")
    raise SystemExit

output_folder = os.path.join(input_folder, "output")
os.makedirs(output_folder, exist_ok=True)
print(f"\nInput:  {input_folder}")
print(f"Output: {output_folder}\n")

# --- Supported image extensions ---
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
image_files = [f for f in sorted(os.listdir(input_folder)) if f.lower().endswith(exts)]

if not image_files:
    print("No image files found in input folder.")
    raise SystemExit

results = []  # To store tank data for Excel

# --- Process each image in folder ---
for file_name in image_files:
    IMAGE_PATH = os.path.join(input_folder, file_name)
    print(f"\n=== Processing {file_name} ===")

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"[ERROR] Could not read {file_name}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS
    )

    final_display = img.copy()

    if circles is None:
        print("No circles detected.")
        continue

    circles = np.uint16(np.around(circles[0]))
    print(f"Detected {len(circles)} circle(s).")

    # To store per-image overlays for final combined image
    tank_overlays = []

    for idx, (x, y, r) in enumerate(circles, start=1):
        cv2.circle(final_display, (x, y), r, (0, 0, 255), 2)
        cv2.circle(final_display, (x, y), 2, (0, 255, 0), 3)

        label = f"Tank {idx}"
        cv2.putText(final_display, label, (x - r, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        x1, x2 = max(0, x - r), min(img.shape[1], x + r)
        y1, y2 = max(0, y - r), min(img.shape[0], y + r)
        tank_crop = img[y1:y2, x1:x2]

        hsv = cv2.cvtColor(tank_crop, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32) / 255.0

        v_blur = cv2.GaussianBlur(v, (3, 3), 0)
        thresh_val = 0.5
        shadow_mask = (v_blur < thresh_val).astype(np.uint8) * 255

        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)

        h_crop, w_crop = shadow_mask.shape
        circle_mask = np.zeros_like(shadow_mask)
        center = (w_crop // 2, h_crop // 2)
        radius = min(r, w_crop // 2, h_crop // 2)
        cv2.circle(circle_mask, center, radius, 255, -1)

        cleaned_mask = cv2.bitwise_and(shadow_mask, circle_mask)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
        final_mask = np.zeros_like(cleaned_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 100:
                final_mask[labels == i] = 255

        shadow_pixels = np.sum(final_mask > 0)
        total_circle_pixels = np.sum(circle_mask > 0)
        shadow_percentage = (shadow_pixels / total_circle_pixels) * 100
        volume_percentage = 100 - shadow_percentage

        print(f"  Tank {idx}: Shadow {shadow_percentage:.2f}% | Volume {volume_percentage:.2f}%")

        # Save results
        results.append({
            "Image Name": file_name,
            "Tank Index": idx,
            "Shadow %": round(shadow_percentage, 2),
            "Volume %": round(volume_percentage, 2)
        })

        # Create overlay for each tank (for saving)
        overlay = cv2.cvtColor(tank_crop, cv2.COLOR_BGR2RGB).copy()
        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 0] = final_mask
        shadow_overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.75, 0)

        mask3 = cv2.merge([circle_mask] * 3)
        shadow_overlay = cv2.bitwise_and(shadow_overlay, mask3) + cv2.bitwise_and(overlay, cv2.bitwise_not(mask3))

        # Save each overlay result
        out_name = f"{os.path.splitext(file_name)[0]}_tank{idx}_overlay.png"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(shadow_overlay, cv2.COLOR_RGB2BGR))

        # Store overlay and its position for later merging
        tank_overlays.append((x1, y1, shadow_overlay))

    # Combine all shadow overlays into full image 
    full_overlay = img.copy()
    for (x1, y1, tank_overlay) in tank_overlays:
        h, w, _ = tank_overlay.shape
        full_overlay[y1:y1+h, x1:x1+w] = cv2.addWeighted(
            full_overlay[y1:y1+h, x1:x1+w], 0.7, cv2.cvtColor(tank_overlay, cv2.COLOR_RGB2BGR), 0.3, 0
        )

    # Save combined overlay
    out_full = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_full_overlay.png")
    cv2.imwrite(out_full, full_overlay)
    print(f"Saved full overlay image -> {out_full}")

    # === Show full overlay ===
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(full_overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Full Shadow Overlay - {file_name}")
    plt.axis('off')
    plt.show()

# --- Save results to Excel ---
if results:
    df = pd.DataFrame(results)
    avg_vol = df["Volume %"].mean()
    df.loc[len(df)] = ["AVERAGE", "-", "-", round(avg_vol, 2)]
    excel_path = os.path.join(output_folder, "Tank_Shadow_Volume_Data.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\nExcel saved: {excel_path}")
    print(f"Average Volume across all tanks: {avg_vol:.2f}%")
else:
    print("\nNo tanks detected in any images. Excel not created.")

print("\nAll done ")
