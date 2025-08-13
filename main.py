# Importing required libraries
import os
import cv2
import numpy as np
import pandas as pd
import time
import torch
from ultralytics import YOLO
from sort import Sort  # Implementation of SORT algorithm (copied from internet)


# Configuring project

VIDEO_PATH = "traffic.mp4"  # Given yt video in the assignment (download it and save it as traffic.mp4 in the root directory)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vehicle_count.csv")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "processed_video.mp4")


# Step1: Device setup 

device = 0 if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {'GPU' if device == 0 else 'CPU'}")

# Step2: Loading YOLO model

print("[INFO] Loading YOLO model...")
model = YOLO("yolov8n.pt")  # Nano model for speed

# Step3: Defining vehicle classes and lanes

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Defining lanes using polygons (adjusted based on the video)
lanes = {
    1: np.array([[100, 300], [500, 300], [500, 600], [100, 600]]),  # Left lane
    2: np.array([[500, 300], [900, 300], [900, 600], [500, 600]]),  # Middle lane
    3: np.array([[900, 300], [1300, 300], [1300, 600], [900, 600]]) # Right lane
}

def get_lane(x, y):
    """Return lane ID if (x, y) is inside a lane polygon."""
    for lane_id, poly in lanes.items():
        if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
            return lane_id
    return None

# Step4: Initialising tracker and data structures

tracker = Sort()
vehicle_data = []  # For CSV
seen = set()       # Avoid duplicate entries
frame_count = 0
lane_counts = {1: 0, 2: 0, 3: 0}

# Step5: Video setup

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

start_time = time.time()
print("[INFO] Processing video... Press 'q' to quit early.")

# Step6: Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # YOLO detection
    results = model(frame, verbose=False, device=device)
    detections = []

    for r in results[0].boxes:
        cls = int(r.cls)
        if cls in vehicle_classes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append([x1, y1, x2, y2, float(r.conf[0])])

    # Tracking
    tracked = tracker.update(np.array(detections)) if len(detections) > 0 else []

    # Draw lanes
    for poly in lanes.values():
        cv2.polylines(frame, [poly], True, (255, 0, 0), 3)

    # Process tracked objects
    for t in tracked:
        x1, y1, x2, y2, track_id = map(int, t)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        lane = get_lane(cx, cy)

        if lane:
            key = (track_id, lane)
            if key not in seen:
                timestamp = round(frame_count / fps, 2)
                vehicle_data.append([track_id, lane, frame_count, timestamp])
                lane_counts[lane] += 1
                seen.add(key)

            # Draw ID and lane on video
            cv2.putText(frame, f"ID:{track_id} L:{lane}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Overlay lane counts
    cv2.putText(frame, f"Lane 1: {lane_counts[1]}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Lane 2: {lane_counts[2]}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Lane 3: {lane_counts[3]}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show and save video
    cv2.imshow("Traffic Analysis", frame)
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step7: Cleaning up and saving results
cap.release()
video_writer.release()
cv2.destroyAllWindows()

df = pd.DataFrame(vehicle_data, columns=["Vehicle_ID", "Lane", "Frame", "Timestamp"])
df.to_csv(OUTPUT_CSV, index=False)

print("\n[INFO] Processing Complete!")
print(f"[INFO] Processed video saved as {OUTPUT_VIDEO}")
print(f"[INFO] CSV saved as {OUTPUT_CSV}")
print("\nVehicle count per lane:")
print(lane_counts)
print(f"\n[INFO] Total time: {round(time.time() - start_time, 2)} seconds")
