import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def analyze_grayscale_threshold_with_volume():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Processing image: {file_path}")
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error: Unable to load image from {file_path}")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect circles
    circles = cv2.HoughCircles(
        blurred_image, cv2.HOUGH_GRADIENT,
        dp=1, minDist=100,
        param1=100, param2=50,
        minRadius=20, maxRadius=70
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        total_tank_volume = 100000  # Example full volume
        processed_rois = []

        for idx, i in enumerate(circles[0, :]):
            x, y, r = i[0], i[1], i[2]

            # Create circular mask
            circle_mask = np.zeros_like(gray_image)
            cv2.circle(circle_mask, (x, y), r, 255, -1)

            # Apply mask to grayscale image
            roi = cv2.bitwise_and(blurred_image, blurred_image, mask=circle_mask)

            # Thresholding (grayscale only)
            _, thresholded_roi = cv2.threshold(roi, 127, 255, cv2.THRESH_TOZERO)

            # Draw circle on original image
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)

            # Calculate volume based on visible pixels
            total_pixels = np.sum(circle_mask == 255)
            visible_pixels = np.sum(thresholded_roi > 0)

            visible_ratio = visible_pixels / total_pixels if total_pixels > 0 else 0
            estimated_volume = total_tank_volume * visible_ratio

            print(f"--- Tank {idx + 1} ---")
            print(f"Center: ({x}, {y}), Radius: {r}")
            print(f"Visible Area %: {visible_ratio * 100:.2f}%")
            print(f"Estimated Volume: {estimated_volume:.2f} (Full Volume: {total_tank_volume})")
            print("-" * 30)

            processed_rois.append((thresholded_roi, f"Tank {idx + 1}"))

        # Display original image with circles
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image with Detected Circles")
        plt.axis('off')
        plt.show()

        # Display all processed tanks
        n = len(processed_rois)
        cols = 3
        rows = (n + cols - 1) // cols

        plt.figure(figsize=(15, 5 * rows))
        for idx, (roi_img, title) in enumerate(processed_rois):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(roi_img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    else:
        print("No circles detected.")

analyze_grayscale_threshold_with_volume()
