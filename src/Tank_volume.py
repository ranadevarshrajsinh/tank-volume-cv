import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def detect_circles_and_display(image_path):
    """
    Loads an image, detects circles with and without histogram equalization,
    and displays the results for comparison.

    Args:
        image_path (str): The full path to the image file.
    """
    # --- 1. Load the Image ---
    # Read the original image from the provided path
    # The image is loaded in BGR (Blue, Green, Red) format by default.
    original_image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if original_image is None:
        print(f"Error: Could not open or find the image at '{image_path}'")
        print("Please check the file path and ensure it is correct.")
        return

    # Convert the image to grayscale for processing
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # --- 2. Create Copies for Displaying Results ---
    # It's important to draw on copies so the original data isn't altered
    output_no_hist = original_image.copy()
    output_with_hist = original_image.copy()

    # --- 3. Process 1: Circle Detection WITHOUT Histogram Equalization ---
    print("\nRunning detection on the standard grayscale image...")
    # Apply a median blur to reduce noise, which is effective against 'salt and pepper' noise
    # and helps prevent false circle detections. A 5x5 kernel is used.
    blurred_gray = cv2.medianBlur(gray_image, 5)

    # Detect circles using the Hough Circle Transform.
    # cv2.HOUGH_GRADIENT: The detection method.
    # dp=1: Inverse ratio of the accumulator resolution to the image resolution. 1 means same resolution.
    # minDist=50: Minimum distance between the centers of detected circles.
    # param1=200: Upper threshold for the internal Canny edge detector.
    # param2=40: Threshold for center detection. Smaller values detect more circles (including false ones).
    # minRadius=20: Minimum circle radius to be detected.
    # maxRadius=100: Maximum circle radius to be detected.
    # NOTE: These parameters are crucial and may need tuning for different images.
    circles_no_hist = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                       param1=200, param2=40, minRadius=20, maxRadius=100)

    # --- 4. Process 2: Circle Detection WITH Histogram Equalization ---
    print("Running detection on the image with adaptive histogram equalization...")
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) is used.
    # It enhances contrast in a more localized way, which is often better than global equalization.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hist_equalized_gray = clahe.apply(gray_image)
    
    # Apply median blur to the equalized image
    blurred_hist_equalized = cv2.medianBlur(hist_equalized_gray, 5)

    # Run Hough Circle Transform on the equalized and blurred image
    circles_with_hist = cv2.HoughCircles(blurred_hist_equalized, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                         param1=200, param2=40, minRadius=20, maxRadius=100)

    # --- 5. Draw Detected Circles on Output Images ---
    # Draw circles found without histogram equalization
    if circles_no_hist is not None:
        # Convert circle parameters (x, y, radius) to integers
        circles_no_hist = np.uint16(np.around(circles_no_hist))
        print(f"Found {len(circles_no_hist[0, :])} circles WITHOUT histogram equalization.")
        for i in circles_no_hist[0, :]:
            # Draw the outer circle (green)
            cv2.circle(output_no_hist, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle (red)
            cv2.circle(output_no_hist, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        print("Found no circles WITHOUT histogram equalization.")

    # Draw circles found with histogram equalization
    if circles_with_hist is not None:
        # Convert circle parameters to integers
        circles_with_hist = np.uint16(np.around(circles_with_hist))
        print(f"Found {len(circles_with_hist[0, :])} circles WITH histogram equalization.")
        for i in circles_with_hist[0, :]:
            # Draw the outer circle (green)
            cv2.circle(output_with_hist, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle (red)
            cv2.circle(output_with_hist, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        print("Found no circles WITH histogram equalization.")

    # --- 6. Display the Results Side-by-Side ---
    # Create a figure to display the plots
    plt.figure(figsize=(18, 10))
    # Use the filename in the suptitle for clarity
    window_title = f"Circle Detection Comparison for: {os.path.basename(image_path)}"
    plt.suptitle(window_title, fontsize=16)

    # Subplot 1: Grayscale image
    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('1. Grayscale Image')
    plt.axis('off')

    # Subplot 2: Histogram Equalized image
    plt.subplot(2, 2, 2)
    plt.imshow(hist_equalized_gray, cmap='gray')
    plt.title('2. Adaptive Histogram Equalization (CLAHE)')
    plt.axis('off')

    # Subplot 3: Result without equalization
    plt.subplot(2, 2, 3)
    # Convert from BGR to RGB for correct color display in matplotlib
    plt.imshow(cv2.cvtColor(output_no_hist, cv2.COLOR_BGR2RGB))
    plt.title('3. Detection Result (Without Equalization)')
    plt.axis('off')

    # Subplot 4: Result with equalization
    plt.subplot(2, 2, 4)
    # Convert from BGR to RGB
    plt.imshow(cv2.cvtColor(output_with_hist, cv2.COLOR_BGR2RGB))
    plt.title('4. Detection Result (With Equalization)')
    plt.axis('off')

    # Show the final plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # --- Set up the root Tkinter window ---
    root = tk.Tk()
    root.withdraw() # Hide the main window

    # --- Open file dialog to select a FOLDER ---
    print("Opening file dialog to select a folder...")
    folder_path = filedialog.askdirectory(title="Select Folder Containing Images")

    # --- Run the detection if a folder was selected ---
    if folder_path: # The path will be an empty string if the user cancels
        print(f"Folder selected: {folder_path}\n")
        
        # Define allowed image extensions
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        # Loop through all files in the selected directory
        for filename in os.listdir(folder_path):
            # Check if the file has a valid image extension
            if filename.lower().endswith(image_extensions):
                # Construct the full path to the image
                image_path = os.path.join(folder_path, filename)
                print(f"--- Processing image: {filename} ---")
                detect_circles_and_display(image_path)
            
        print("\n--- All images in the folder have been processed. ---")

    else:
        print("No folder selected. Exiting program.")

## For  Histogram Equalization.