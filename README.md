# YOLO Stone Tracker

Real-time object detection and tracking with trajectory visualization.

## Quick Start

### Local Development

```bash
# Using Python's built-in server
cd yolo-tracker
python3 -m http.server 8080
```

Then open http://localhost:8080

### Deploy to GitHub Pages / Vercel

1. Push to GitHub:
```bash
cd ~/Workspace/Projects/yolo-tracker
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/yolo-tracker.git
git push -u origin main
```

2. Deploy:
   - **GitHub Pages**: Settings → Pages → Deploy from main branch
   - **Vercel**: Import project from GitHub → Deploy

## Features

- Real-time webcam detection using YOLOv8
- Click and drag to select and name objects
- Multi-object tracking with unique colors
- Trajectory visualization (20 min history)
- Trajectory point filtering for memory optimization

## Usage

1. Click **Start** to enable camera
2. **Click and drag** to draw a box around a stone
3. Enter a name for the object
4. The object will be tracked with a colored box and trajectory line
5. **Double-click** on a tracked object to remove it
6. Click **Clear All** to remove all tracked objects

## Technical Details

- Model: YOLOv8n (nano) - ~6M parameters
- Runtime: ONNX Runtime Web (WASM)
- Input: 640x640 normalized
- Output: 8400 detections with 80 classes (COCO)
- Detection interval: 100ms (10 FPS)
- Trajectory: Points filtered by distance (>10px) and time (>500ms)
- Max trajectory duration: 20 minutes
