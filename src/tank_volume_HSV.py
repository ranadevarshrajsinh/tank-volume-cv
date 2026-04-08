import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

def detect_circles_and_display(image_path):
    """
    Loads an image, detects circles using standard Grayscale vs. the HSV Value channel,
    and displays the results for comparison.

    Args:
        image_path (str): The full path to the image file.
    """
    # --- 1. Load the Image ---
    # Read the original image from the provided path
    original_image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if original_image is None:
        print(f"Error: Could not open or find the image at '{image_path}'")
        print("Please check the file path and ensure it is correct.")
        return

    # --- 2. Create Copies for Displaying Results ---
    output_from_gray = original_image.copy()
    output_from_hsv = original_image.copy()

    # --- 3. Process 1: Circle Detection from standard Grayscale (derived from RGB) ---
    print("\nRunning detection on the standard grayscale image...")
    # Convert the image to grayscale for processing
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Apply a median blur to reduce noise
    blurred_gray = cv2.medianBlur(gray_image, 5)

    # Detect circles using the Hough Circle Transform on the grayscale image
    # dp: Inverse ratio of accumulator resolution. 1 is the same as original.
    # minDist: Minimum distance between centers of detected circles.
    # param1: Upper threshold for the internal Canny edge detector.
    # param2: Accumulator threshold for circle center detection. Lower means more circles will be detected (including false positives).
    # minRadius/maxRadius: The range of circle sizes to look for.
    circles_from_gray = cv2.HoughCircles(blurred_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                         param1=195, param2=45, minRadius=20, maxRadius=100)

    # --- 4. Process 2: Circle Detection using HSV Value Channel ---
    print("Running detection on the HSV Value channel...")
    # Convert the original BGR image to HSV
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    # The 'V' (Value) channel often isolates brightness information effectively.
    # We extract just this channel for our second detection method.
    value_channel = hsv_image[:, :, 2]
    
    # Apply median blur to the value channel
    blurred_value_channel = cv2.medianBlur(value_channel, 5)

    # Run Hough Circle Transform on the blurred value channel with adjusted parameters.
    # We lower param2 to make the detection more sensitive and potentially find more circles.
    circles_from_hsv = cv2.HoughCircles(blurred_value_channel, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                        param1=195, param2=40, minRadius=20, maxRadius=100)

    # --- 5. Draw Detected Circles on Output Images ---
    # Draw circles found from the grayscale image
    if circles_from_gray is not None:
        circles_from_gray = np.uint16(np.around(circles_from_gray))
        print(f"Found {len(circles_from_gray[0, :])} circles from Grayscale.")
        for i in circles_from_gray[0, :]:
            cv2.circle(output_from_gray, (i[0], i[1]), i[2], (0, 255, 0), 2) # Circle
            cv2.circle(output_from_gray, (i[0], i[1]), 2, (0, 0, 255), 3)   # Center
    else:
        print("Found no circles from Grayscale.")

    # Draw circles found from the HSV value channel
    if circles_from_hsv is not None:
        circles_from_hsv = np.uint16(np.around(circles_from_hsv))
        print(f"Found {len(circles_from_hsv[0, :])} circles from HSV Value Channel.")
        for i in circles_from_hsv[0, :]:
            cv2.circle(output_from_hsv, (i[0], i[1]), i[2], (0, 255, 0), 2) # Circle
            cv2.circle(output_from_hsv, (i[0], i[1]), 2, (0, 0, 255), 3)   # Center
    else:
        print("Found no circles from HSV Value Channel.")

    # --- 6. Display the Results Side-by-Side ---
    plt.figure(figsize=(18, 10))
    window_title = f"Circle Detection Comparison for: {os.path.basename(image_path)}"
    plt.suptitle(window_title, fontsize=16)

    # Subplot 1: Grayscale image
    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('1. Grayscale Image (from RGB)')
    plt.axis('off')

    # Subplot 2: HSV Value Channel
    plt.subplot(2, 2, 2)
    plt.imshow(value_channel, cmap='gray')
    plt.title('2. HSV Value Channel')
    plt.axis('off')

    # Subplot 3: Result from Grayscale
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(output_from_gray, cv2.COLOR_BGR2RGB))
    plt.title('3. Detection Result (from Grayscale)')
    plt.axis('off')

    # Subplot 4: Result from HSV
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(output_from_hsv, cv2.COLOR_BGR2RGB))
    plt.title('4. Detection Result (from HSV, param1=190, param2=45)')
    plt.axis('off')

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

