# Traffic Flow Analysis

## 📌 Objective
The goal of this project is to analyze traffic flow using computer vision techniques with the following objectives:
- Detect and track vehicles in a video feed.
- Assign vehicles to **curved lanes** using polygonal regions.
- Maintain **separate vehicle counts for each lane**.
- Generate:
  - A **processed video** with overlays showing vehicle IDs, assigned lanes, and lane-wise counts.
  - A **CSV file** containing detailed data (Vehicle ID, Lane, Frame, Timestamp).

---

## ✅ Features
- **Vehicle Detection**: Utilizes **YOLOv8** (pre-trained on COCO dataset) for accurate vehicle detection.
- **Lane Detection**: Implements polygon-based lane detection to handle **curved roads**.
- **Vehicle Tracking**: Employs the **SORT (Simple Online and Realtime Tracking) algorithm** to ensure vehicles are not counted multiple times.
- **Real-Time Overlay**:
  - Displays vehicle IDs and their assigned lanes on the video.
  - Shows lane-wise vehicle counts in real-time.
- **CSV Output**: Generates a CSV file with the following columns:
  - Vehicle ID
  - Lane Number
  - Frame Number
  - Timestamp (in seconds)
- **Processed Video**: Saves the annotated video for demonstration purposes.
- **Automatic GPU Detection**: Automatically detects and uses GPU if available, with fallback to CPU.

---

## 📂 Project Structure
```
traffic-flow-analysis/
├── main.py                 # Main script for running the analysis
├── sort.py                 # Implementation of the SORT tracking algorithm
├── outputs/                # Directory for output files
│   ├── vehicle_count.csv   # CSV file with vehicle tracking data
│   ├── processed_video.mp4 # Processed video with annotations
├── traffic.mp4             # Input traffic video (download it manually)
├── requirements.txt        # List of project dependencies
├── README.md               # Project documentation
├── .gitignore              # gitignore file for github
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
Clone the project repository to your local machine:
```bash
git clone <your-repo-url>
cd traffic-flow-analysis
```

### 2. Create a Virtual Environment
Python 3.11 is recommended (YOLOv8 does not yet support Python 3.12).
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```
OR

If you have Anaconda, run:
```bash
conda create -n traffic python=3.11 -y
conda activate traffic
```

### 3. Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### `requirements.txt` Contents
```
ultralytics
opencv-python
pandas
filterpy
torch
```

### 4. Download the Input Video
Manually download the traffic video from the following source:
- **YouTube**: [Traffic Video](<https://www.youtube.com/watch?v=MNn9qKG2UFI>)
- Save the downloaded video as `traffic.mp4` in the project root directory.

---

## ▶️ Running the Project
Run the main script to process the traffic video:
```bash
python main.py
```

- **Controls**: Press `q` to quit the processing early.
- If not interrupted, the script will process the entire video.

---

## ✅ Outputs
Upon successful execution, the following outputs will be generated:
- **Processed Video**: Saved as `outputs/processed_video.mp4` with vehicle IDs, lane assignments, and lane-wise counts overlaid.
- **CSV File**: Saved as `outputs/vehicle_count.csv` containing detailed vehicle tracking data.
- **Console Summary**: Displays a summary of vehicle counts per lane, e.g.:
  ```yaml
  Vehicle count per lane:
  Lane 1: 561
  Lane 2: 402
  Lane 3: 60
  ```