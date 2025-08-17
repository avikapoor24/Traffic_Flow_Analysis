import sys, platform
import os, math, time, json
from datetime import timedelta
import numpy as np
import pandas as pd
import cv2
from IPython.display import display, HTML

print('Python', sys.version)
print('Platform', platform.platform())
!nvidia-smi || echo 'No GPU available, continuing on CPU.'

# Install dependencies
!pip install yt-dlp
!pip install ultralytics opencv-python pytube pandas filterpy
!pip install lapx
print('✅ Dependencies installed')



VEHICLE_CLASS_IDS = {2:'car', 3:'motorbike', 5:'bus', 7:'truck'}  # COCO IDs

def draw_lanes(frame, lanes):
    for i, poly in enumerate(lanes, start=1):
        pts = np.array(poly, np.int32)
        cv2.polylines(frame, [pts], True, (0,255,0), 2)
        # Put lane label near the first point
        x,y = pts[0]
        cv2.putText(frame, f'Lane {i}', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    return frame

def point_in_lane(point, lane_polygon):
    # point: (x,y)
    return cv2.pointPolygonTest(np.array(lane_polygon, np.int32), point, False) >= 0

def mmss_from_frame(frame_idx, fps):
    if fps <= 0:
        return '00:00'
    seconds = frame_idx / fps
    return str(timedelta(seconds=int(seconds)))

# Download the video via yt-dlp (more reliable than pytube). Adjust URL if needed.
VIDEO_URL = 'https://www.youtube.com/watch?v=MNn9qKG2UFI'
VIDEO_PATH = 'traffic.mp4'

if not os.path.exists(VIDEO_PATH):
    !yt-dlp -f mp4 -o "{VIDEO_PATH}" {VIDEO_URL}
else:
    print('Video already downloaded:', VIDEO_PATH)
assert os.path.exists(VIDEO_PATH), 'Video download failed — check the URL or rerun this cell.'
print('✅ Video ready:', VIDEO_PATH)

# Grab and display the first frame to help you choose lane coordinates
cap = cv2.VideoCapture('traffic.mp4')
ok, frame = cap.read()
cap.release()
assert ok, 'Could not read first frame. Try re-downloading the video.'
cv2.imwrite('first_frame.jpg', frame)
print('Saved first_frame.jpg — download it, annotate lanes, and set coordinates in the next cell.')
display(HTML('<img src="first_frame.jpg" width="900">'))

# ✍️ EDIT THIS: Define your 3 lane polygons as lists of (x,y) pixel coordinates.
# Tip: Place points clockwise around each lane area.
lanes = [
    [(35,270), (230,270), (230,340), (35,340)],   # Lane 1
    [(260,270), (360,270), (360,340), (260,340)], # Lane 2
    [(400,270), (600,270), (600,340), (400,340)]  # Lane 3
]


print('Lane polygons set to:', lanes)

# Visual check on the first frame
img = cv2.imread('first_frame.jpg')
img = draw_lanes(img.copy(), lanes)
cv2.imwrite('lanes_preview.jpg', img)
display(HTML('<img src="lanes_preview.jpg" width="900">'))

# Remove old SORT folder if exists
!rm -rf sort

# Clone SORT repo
!git clone -q https://github.com/abewley/sort.git

!sed -i '/matplotlib/d' sort/sort.py
!sed -i '/TkAgg/d' sort/sort.py

from sort.sort import Sort
print('✅ SORT ready (no matplotlib dependencies)')

# Load YOLOv8 (nano for speed)
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('✅ YOLO model loaded')

cap = cv2.VideoCapture('traffic.mp4')
assert cap.isOpened(), 'Cannot open video file.'

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Video: {width}x{height} @ {fps:.2f} FPS, frames: {total_frames}')

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, max(10.0, fps if fps>0 else 25.0), (width, height))

tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.2)
counts = {1:0, 2:0, 3:0}
counted_pairs = set()  # (track_id, lane_id)
vehicle_log = []  # rows: [VehicleID, Lane, Frame, Timestamp]

frame_idx = 0
start_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1

    # Inference
    res = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]

    dets = []  # [x1,y1,x2,y2,score]
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        for (x1,y1,x2,y2), sc, cls in zip(boxes, scores, classes):
            if cls in VEHICLE_CLASS_IDS:
                dets.append([x1, y1, x2, y2, float(sc)])

    dets_np = np.array(dets) if len(dets) else np.empty((0,5))
    tracks = tracker.update(dets_np)  # each: [x1,y1,x2,y2,ID]

    # Draw lanes
    draw_lanes(frame, lanes)

    # Process tracks
    for tr in tracks:
        x1,y1,x2,y2,tid = tr
        x1,y1,x2,y2,tid = float(x1), float(y1), float(x2), float(y2), int(tid)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Lane membership & counting
        for lane_id, poly in enumerate(lanes, start=1):
            if point_in_lane((cx, cy), poly):
                if (tid, lane_id) not in counted_pairs:
                    counts[lane_id] += 1
                    counted_pairs.add((tid, lane_id))
                # Log every frame that a track is in a lane. To reduce CSV size, log only first time:
                if not any((r[0]==tid and r[1]==lane_id) for r in vehicle_log):
                    vehicle_log.append([tid, lane_id, frame_idx, mmss_from_frame(frame_idx, fps)])
                break

        # Drawing track box & ID
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
        cv2.circle(frame, (cx,cy), 3, (255,255,255), -1)
        cv2.putText(frame, f'ID {tid}', (int(x1), int(y1)-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Overlay counts
    for lid in (1,2,3):
        cv2.putText(frame, f'Lane {lid}: {counts[lid]}', (20, 40*lid), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    out.write(frame)

    if frame_idx % 200 == 0:
        elapsed = time.time() - start_time
        fps_est = frame_idx / max(1e-6, elapsed)
        print(f'Processed {frame_idx}/{total_frames} frames (~{fps_est:.1f} FPS)')

cap.release()
out.release()

print('✅ Processing complete')
print('Final counts:', counts)

# Save CSV
df = pd.DataFrame(vehicle_log, columns=['VehicleID','Lane','Frame','Timestamp'])
df.to_csv('traffic_count.csv', index=False)
print('Saved traffic_count.csv with', len(df), 'rows')

# Preview CSV (first 10 rows)
import pandas as pd
df = pd.read_csv('traffic_count.csv')
df.head(10)

# Embed the output video for a quick preview
from base64 import b64encode
video_path = 'output.mp4'
if os.path.exists(video_path):
    mp4 = open(video_path,'rb').read()
    data_url = 'data:video/mp4;base64,' + b64encode(mp4).decode()
    display(HTML(f'<video controls width=900 src="{data_url}"></video>'))
else:
    print('output.mp4 not found.')

# Save outputs to Google Drive
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/traffic_flow_output
!cp output.mp4 /content/drive/MyDrive/traffic_flow_output/
!cp traffic_count.csv /content/drive/MyDrive/traffic_flow_output/
print('✅ Saved to /content/drive/MyDrive/traffic_flow_output/')
