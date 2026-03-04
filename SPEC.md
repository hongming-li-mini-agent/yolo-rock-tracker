# YOLO Stone Tracker - Specification

## Project Overview
- **Name**: yolo-tracker
- **Type**: Web Application (Single HTML + JS)
- **Core Functionality**: Real-time object detection and tracking on webcam feed with trajectory visualization
- **Target Users**: Desktop users who want to track multiple stones/objects via webcam

## Technical Stack
- **Frontend**: Vanilla HTML/JS (no framework)
- **Model**: YOLOv8n ONNX (pretrained, detects "rock" as class)
- **Runtime**: ONNX Runtime Web (WASM)
- **Deployment**: GitHub Pages / Vercel

## UI/UX Specification

### Layout Structure
- **Full-screen canvas** overlaid on webcam video
- **Control panel** (top-right): Start/Stop, Clear, Model status
- **Object list** (bottom-left): Shows tracked objects with names

### Visual Design
- **Background**: Webcam feed (full viewport)
- **Box color**: Each object gets unique color from palette
- **Box style**: 2px solid border, semi-transparent fill (10% opacity)
- **Label**: Object name + confidence, positioned at top-left of box
- **Trajectory**: 2px solid line in matching color, 20min max history
- **Font**: JetBrains Mono, 14px for labels

### Color Palette
```javascript
const COLORS = [
  '#FF6B6B', // Red
  '#4ECDC4', // Teal
  '#FFE66D', // Yellow
  '#95E1D3', // Mint
  '#F38181', // Coral
  '#AA96DA', // Purple
  '#FCBAD3', // Pink
  '#A8D8EA', // Sky
];
```

### Interactions
1. **Click + Drag**: Draw initial bounding box to select area
2. **Release**: Prompt for object name
3. **Double-click** on tracked object: Remove tracking

## Functionality Specification

### Core Features

1. **Webcam Access**
   - Request camera permission on load
   - Mirror video horizontally (natural interaction)
   - Handle camera errors gracefully

2. **Object Detection (YOLOv8)**
   - Run inference every 100ms (10 FPS for detection)
   - Filter for "rock" class (class 0 in COCO, but we'll use all and let user pick)
   - Confidence threshold: 0.5

3. **Object Tracking**
   - Initial selection: User draws box → assigns name
   - Track by: Intersection-over-union (IoU) matching with detections
   - If no match: Hide box, keep waiting for re-detection
   - Each object stores: id, name, color, trajectory[], lastSeen

4. **Trajectory Management**
   - Store points as [x, y, timestamp]
   - Filter: Only add point if distance > 10px AND time > 500ms from last point
   - Max duration: 20 minutes (clear older points)
   - Draw as polyline connecting all points

5. **Detection Results**
   - Auto-create tracking for detections with >0.7 confidence
   - User can click to convert any detection to tracked object

### Data Handling
- All processing client-side (no server)
- Trajectory stored in memory (not persisted)
- ONNX model loaded from CDN or local

### Edge Cases
- Camera denied: Show error message
- No detection: Show "No objects detected" status
- Multiple overlapping detections: Track each independently
- Object exits frame: Hide box, keep trying to re-acquire

## File Structure
```
yolo-tracker/
├── index.html      # Main HTML
├── tracker.js      # Core logic
├── style.css       # Styles
├── yolov8n.onnx    # Model (downloaded)
└── README.md       # Setup instructions
```

## Acceptance Criteria

- [ ] Webcam stream displays (mirrored)
- [ ] YOLOv8 loads and runs inference
- [ ] Detections shown as boxes in real-time
- [ ] User can draw box to create new tracked object
- [ ] Each object has unique color and label
- [ ] Trajectory line draws behind moving objects
- [ ] Trajectory filters points (no excessive memory)
- [ ] Multiple objects can be tracked simultaneously
- [ ] Works on desktop Chrome/Firefox/Safari
