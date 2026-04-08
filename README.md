# Tank Volume Estimation using Image Processing

![Tank Detection Example](docs/assets/Screenshot%202025-09-06%20004327.png)

An automated solution for detecting storage tanks and estimating their liquid volume based on shadow analysis. This project uses computer vision techniques to identify circular tanks from overhead/side imagery and calculates the shadow percentage to infer the current volume level.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Computer Vision**: OpenCV
- **Data Analysis**: Pandas, NumPy
- **Excel Export**: OpenPyXL
- **Visualization**: Matplotlib
- **GUI**: Tkinter (for folder selection)

## 🚀 Features

- **Automated Tank Detection**: Uses Hough Circle Transform to identify circular tanks in images.
- **Shadow Analysis**: Extracts shadows using HSV/Value channel analysis to differentiate between filled and empty regions.
- **Volume Estimation**: Calculates the percentage of liquid volume based on detected shadow area.
- **Batch Processing**: Supports processing multiple images from a folder.
- **Data Export**: Generates annotated images and a comprehensive Excel report (`Tank_Shadow_Volume_Data.xlsx`) with volume percentages and averages.
- **Preprocessing Comparison**: Includes tools to compare circle detection with and without Adaptive Histogram Equalization (CLAHE).

## 📂 Project Structure

```text
tank-volume-cv/
├── src/                    # Source code scripts
│   ├── v_final.py          # Main script for tank detection & volume calculation
│   ├── Tank_volume.py      # Comparison tool for circle detection methods
│   ├── volume_HSV.py       # HSV-based volume estimation logic
│   └── ...                 # Experimental and older script versions
├── data/
│   ├── oil_tank_dataset/    # Dataset containing tank comparison data (Feb/Nov 2020)
│   └── samples/            # Sample tank images for testing
├── docs/
│   ├── assets/             # Screenshots and documentation visual aids
│   └── Tank_Volume_Analysis_Report.pdf # Full project report and methodology
├── requirements.txt        # Project dependencies
└── README.md
```

## 📄 Documentation

A comprehensive project report is available in the `docs/` folder:
- **[Tank_Volume_Analysis_Report.pdf](docs/Tank_Volume_Analysis_Report.pdf)**: Detailed explanation of the methodology, algorithms used, and analysis of results.

## 🛠️ Installation

1. Clone this repository or download the project.
2. Ensure you have Python 3.x installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Main Volume Estimation
To process images and get volume estimates:
```bash
python src/v_final.py
```
- Select the folder containing your tank images when prompted.
- Select the output folder for results.
- The script will generate annotated images showing detected tanks and their volume percentages.

### 2. Circle Detection Comparison
To see how different preprocessing techniques affect tank detection:
```bash
python src/Tank_volume.py
```

## 📊 How it Works

1. **Preprocessing**: The image is converted to grayscale and blurred to reduce noise.
2. **Detection**: Hough Circle Transform identifies the boundaries of the tanks.
3. **Shadow Extraction**: Within each detected circle, a local background illumination is estimated. Pixels significantly darker than the background are identified as shadows.
4. **Refinement**: Morphological operations (Closing/Opening) and connected components analysis clean up the shadow mask.
5. **Calculation**:
   - `Shadow % = (Shadow Pixels / Total Circle Pixels) * 100`
   - `Volume % = 100 - Shadow %`
   - *Note: Tanks with ≤12% shadow are automatically considered 100% full.*

## 📸 Samples & Results

The system provides visual feedback by overlaying the detected shadow area (in red) and the calculated volume percentage on each tank.

*(Refer to `docs/assets/` for visual results and screenshots of the detection in action)*

## 📄 License
This project is for educational/assignment purposes.

---

## 🔗 Connect with me
- **LinkedIn**: [https://www.linkedin.com/in/devarshrajsinh-rana-7203a6344]
- **Email**: [ranadevarshrajsinh@gmail.com]
