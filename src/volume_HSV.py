import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# ---------------- CONFIG (tweak if needed) ----------------
# Keep Hough parameters you requested
HOUGH_DP = 1.0
HOUGH_MIN_DIST = 75
HOUGH_PARAM1 = 195
HOUGH_PARAM2 = 45
HOUGH_MIN_RADIUS = 20
HOUGH_MAX_RADIUS = 120

# Fill / morphology and thresholds
DRAW_ALPHA = 0.5              # overlay alpha for visualization
MORPH_RELATIVE = 0.04        # kernel size relative to radius for closing/opening
FILL_KERNEL_MIN = 3

# Filters to avoid false positives on fully-filled tanks
FULL_BRIGHT_MEAN = 155       # mean V above this and uniform -> treat as full (no shadow)
FULL_BRIGHT_STD = 15

# Shadow acceptance thresholds
MIN_SHADOW_AREA_FRAC = 0.121  # minimal area fraction of circle to consider (12%)
RELAX_MIN_SHADOW_AREA_FRAC = 0.012
MIN_NEAR_EDGE_FRAC = 0.15    # fraction of shadow contour points near rim
RELAX_NEAR_EDGE_FRAC = 0.10
MIN_OFFSET_FRAC = 0.08       # centroid offset from center (as fraction of r)
RELAX_OFFSET_FRAC = 0.05
MIN_CONTRAST = 18            # difference between avg circle brightness and average shadow brightness

OUT_SUFFIX = "_shadow_output.png"

# ---------------- Helper utilities ----------------
def safe_percentile(arr, q, default=0):
    try:
        return float(np.percentile(arr, q))
    except Exception:
        return default

# ---------------- Core function ----------------
def detect_and_calculate_shadow_percentage(image_path, out_folder):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERR] Cannot open {image_path}")
        return

    h_img, w_img = img.shape[:2]
    output_image = img.copy()

    # Use V channel for circle detection (value/brightness)
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_full = hsv_full[:, :, 2]
    v_blur_for_hough = cv2.medianBlur(v_full, 5)

    # Hough circles (params kept as requested)
    circles = cv2.HoughCircles(
        v_blur_for_hough,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS
    )

    if circles is None:
        print(f"No tanks detected in {os.path.basename(image_path)}")
        return

    circles = np.around(circles[0, :]).astype(int)
    print(f"Processing {os.path.basename(image_path)} -> found {len(circles)} circles")


    for idx, (cx, cy, r) in enumerate(circles, start=1):
        cx, cy, r = int(cx), int(cy), int(r)
        if r <= 4:
            continue

        # Crop safely around circle
        x1 = max(cx - r, 0); y1 = max(cy - r, 0)
        x2 = min(cx + r, w_img - 1); y2 = min(cy + r, h_img - 1)
        crop_bgr = img[y1:y2+1, x1:x2+1].copy()
        if crop_bgr.size == 0:
            print(f"  Tank {idx}: empty crop, skipping")
            continue

        h_crop, w_crop = crop_bgr.shape[:2]
        cxc = cx - x1; cyc = cy - y1
        r_cropped = int(min(r, cxc, cyc, w_crop - 1 - cxc, h_crop - 1 - cyc))
        if r_cropped <= 4:
            print(f"  Tank {idx}: too small after crop, skipping")
            continue

        # Circle mask (correct center = cxc,cyc)
        mask_full = np.zeros((h_crop, w_crop), dtype=np.uint8)
        cv2.circle(mask_full, (cxc, cyc), r_cropped, 255, -1)
        circle_area_px = int(np.count_nonzero(mask_full))
        if circle_area_px == 0:
            continue

        # V channel inside crop (brightness)
        hsv_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        v_crop = hsv_crop[:, :, 2]

        # Mean & std brightness inside circle -> skip fully bright uniform tanks
        circle_vals = v_crop[mask_full == 255]
        mean_brightness = float(np.mean(circle_vals)) if circle_vals.size else 0.0
        std_brightness = float(np.std(circle_vals)) if circle_vals.size else 0.0
        if mean_brightness >= FULL_BRIGHT_MEAN and std_brightness <= FULL_BRIGHT_STD:
            print(f"  Tank {idx}: full/bright (mean={mean_brightness:.1f}, std={std_brightness:.1f}) → SKIP")
            continue

        # Estimate local background illumination by heavy blur, then compute how much darker pixel is than bg
        sigma = max(1.0, r_cropped * 0.5)
        bg = cv2.GaussianBlur(v_crop, (0, 0), sigmaX=sigma)  # background brightness
        diff_img = cv2.subtract(bg, v_crop)  # positive where v_crop is darker than bg

        # Dynamic threshold on the diff image using percentile within circle
        diff_vals = diff_img[mask_full == 255]
        if diff_vals.size == 0:
            print(f"  Tank {idx}: no diff values, skip")
            continue
        thr_val = safe_percentile(diff_vals, 68, default=10)  # 68th percentile (adaptive)
        thr_val = max(8, int(round(thr_val)))  # floor to reasonable minimum

        # Candidate shadow mask (only inside circle)
        shadow_mask = np.zeros_like(diff_img, dtype=np.uint8)
        shadow_mask[(diff_img >= thr_val) & (mask_full == 255)] = 255

        # Morphological cleaning sized relative to radius
        kern_size = max(FILL_KERNEL_MIN, int(round(r_cropped * MORPH_RELATIVE)))
        if kern_size % 2 == 0:
            kern_size += 1
        morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_size, kern_size))
        shadow_clean = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, morph_k, iterations=2)
        shadow_clean = cv2.morphologyEx(shadow_clean, cv2.MORPH_OPEN, morph_k, iterations=1)

        # Find connected components (contours)
        contours, _ = cv2.findContours(shadow_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"  Tank {idx}: no dark region found")
            continue

        # Pick the largest contiguous dark region
        main_shadow = max(contours, key=cv2.contourArea)
        main_area = cv2.contourArea(main_shadow)
        area_frac = main_area / float(circle_area_px)

        # Quick area filter
        if area_frac < MIN_SHADOW_AREA_FRAC:
            # try relaxed fallback: combine close components (if many) or accept if contrast strong and near edge
            # We'll compute small stats and allow a relaxed path if contrast large
            combined_area_frac = area_frac
            # If too tiny, skip for now
            if combined_area_frac < RELAX_MIN_SHADOW_AREA_FRAC:
                print(f"  Tank {idx}: shadow too small (area_frac={area_frac:.3f}), skip")
                continue

        # compute centroid of main shadow
        M = cv2.moments(main_shadow)
        if M.get("m00", 0) == 0:
            print(f"  Tank {idx}: shadow moment zero, skip")
            continue
        sx = int(M["m10"] / M["m00"]); sy = int(M["m01"] / M["m00"])
        offset_ratio = np.hypot((sx - cxc), (sy - cyc)) / float(r_cropped)

        # compute how many contour points are near the rim (strong sign of crescent)
        pts = main_shadow.reshape(-1, 2)
        dists = np.sqrt((pts[:, 0] - cxc) ** 2 + (pts[:, 1] - cyc) ** 2)
        near_edge_count = int(np.count_nonzero(dists >= (0.85 * r_cropped)))
        proportion_near_edge = near_edge_count / float(max(1, len(pts)))

        # contrast check: brightness difference between circle mean and dark region mean
        dark_vals = v_crop[shadow_clean == 255]
        if dark_vals.size == 0:
            print(f"  Tank {idx}: dark region has no pixels, skip")
            continue
        mean_dark = float(np.mean(dark_vals))
        contrast = mean_brightness - mean_dark

        # Acceptance logic (strict OR relaxed)
        accept = False
        mode = None
        if (area_frac >= MIN_SHADOW_AREA_FRAC and
            (proportion_near_edge >= MIN_NEAR_EDGE_FRAC or offset_ratio >= MIN_OFFSET_FRAC) and
            contrast >= MIN_CONTRAST):
            accept = True
            mode = "strict"
        else:
            # relaxed fallback
            if (area_frac >= RELAX_MIN_SHADOW_AREA_FRAC and
                (proportion_near_edge >= RELAX_NEAR_EDGE_FRAC or offset_ratio >= RELAX_OFFSET_FRAC) and
                contrast >= (MIN_CONTRAST - 6)):
                accept = True
                mode = "relaxed"

        if not accept:
            print(f"  Tank {idx}: rejected (area={area_frac:.3f}, offset={offset_ratio:.3f}, edge={proportion_near_edge:.3f}, contrast={contrast:.1f})")
            continue

        # Build selected_mask from main_shadow (and optionally dilate & floodfill to fill gaps)
        selected_mask = np.zeros_like(shadow_clean)
        cv2.drawContours(selected_mask, [main_shadow], -1, 255, -1)

        # Dilate a little to connect close fragments then floodfill from centroid
        fill_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, kern_size), max(3, kern_size)))
        dilated = cv2.dilate(selected_mask, fill_k, iterations=2)

        # seed for floodfill -> prefer centroid of main_shadow (sx,sy); ensure inside bounds
        seed_x = np.clip(sx, 0, w_crop - 1)
        seed_y = np.clip(sy, 0, h_crop - 1)
        if dilated[seed_y, seed_x] == 0:
            nz = np.transpose(np.nonzero(selected_mask))
            if nz.shape[0] > 0:
                # pick first pixel inside region
                seed_y, seed_x = int(nz[0][0]), int(nz[0][1])
            else:
                seed_x, seed_y = cxc, cyc

        flood = dilated.copy()
        flood_mask = np.zeros((h_crop + 2, w_crop + 2), dtype=np.uint8)
        try:
            cv2.floodFill(flood, flood_mask, (seed_x, seed_y), 255)
        except Exception:
            # If floodFill fails, keep dilated as fallback
            flood = dilated

        # keep only inside circle
        flood = cv2.bitwise_and(flood, mask_full)
        # combine original selection with flood and close tiny holes
        filled_shadow = cv2.bitwise_or(selected_mask, flood)
        filled_shadow = cv2.morphologyEx(filled_shadow, cv2.MORPH_CLOSE, fill_k, iterations=1)

        # remove tiny specks with connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled_shadow, connectivity=8)
        if num_labels > 1:
            min_comp_area = max(8, int(0.001 * circle_area_px))
            cleaned = np.zeros_like(filled_shadow)
            for lbl in range(1, num_labels):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                if area >= min_comp_area:
                    cleaned[labels == lbl] = 255
            filled_shadow = cleaned

        final_shadow_pixels = int(np.count_nonzero(filled_shadow))
        shadow_percentage = (final_shadow_pixels / float(circle_area_px)) * 100.0
        shadow_percentage = min(100.0, shadow_percentage)

        # Visualization: overlay filled_shadow (red) only inside crop & inside circle
        overlay_crop = crop_bgr.copy()
        overlay_color = np.zeros_like(overlay_crop)
        overlay_color[:, :] = (0, 0, 255)
        sel_mask_3c = cv2.merge([filled_shadow, filled_shadow, filled_shadow])
        blend = cv2.addWeighted(overlay_crop, 1 - DRAW_ALPHA, overlay_color, DRAW_ALPHA, 0)
        overlay_crop = np.where(sel_mask_3c == 255, blend, overlay_crop).astype(np.uint8)

        # write back overlay
        output_image[y1:y2+1, x1:x2+1] = overlay_crop

        # draw circle and text (global coordinates)
        cv2.circle(output_image, (cx, cy), r, (0, 255, 0), 2)
        text_pos = (max(cx - r, 5), max(cy - r - 10, 15))
        cv2.putText(output_image, f"{shadow_percentage:.1f}%", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        print(f"  Tank {idx}: ACCEPTED ({mode}) area={area_frac:.3f} offset={offset_ratio:.3f} edge={proportion_near_edge:.3f} contrast={contrast:.1f} -> {shadow_percentage:.1f}%")

    # Save output image into out_folder
    os.makedirs(out_folder, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_folder, base + OUT_SUFFIX)
    cv2.imwrite(out_path, output_image)
    print(f"Saved -> {out_path}")


# ---------------- Main (folder selection) ----------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    print("Select INPUT folder with images.")
    input_folder = filedialog.askdirectory(title="Select INPUT folder (images)")
    if not input_folder:
        print("No input folder selected — exiting.")
        raise SystemExit

    print("Select OUTPUT folder to save annotated results (or Cancel to create 'shadow_output' inside input).")
    output_folder = filedialog.askdirectory(title="Select OUTPUT folder (or Cancel)")
    if not output_folder:
        output_folder = os.path.join(input_folder, "shadow_output")
        os.makedirs(output_folder, exist_ok=True)

    print(f"\nInput:  {input_folder}")
    print(f"Output: {output_folder}\n")

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in sorted(os.listdir(input_folder)) if f.lower().endswith(exts) and not f.lower().endswith(OUT_SUFFIX)]

    if not files:
        print("No images found in input folder. Exiting.")
        raise SystemExit

    for fname in files:
        path = os.path.join(input_folder, fname)
        print(f"\n=== {fname} ===")
        detect_and_calculate_shadow_percentage(path, output_folder)

    print("\nAll done.")
