// YOLO Stone Tracker
// Real-time object detection and tracking with trajectory visualization

const COLORS = [
    '#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3',
    '#F38181', '#AA96DA', '#FCBAD3', '#A8D8EA',
    '#C7F9CC', '#FF9F1C', '#2EC4B6', '#E71D36'
];

// Configuration
const CONFIG = {
    confidenceThreshold: 0.5,
    iouThreshold: 0.3,
    detectionInterval: 100, // ms between detections
    trajectoryMinDistance: 10, // min pixels between points
    trajectoryMinTime: 500, // min ms between points
    trajectoryMaxDuration: 20 * 60 * 1000, // 20 minutes
    boxLineWidth: 2,
    boxFillAlpha: 0.1,
    trajectoryLineWidth: 2
};

// State
let session = null;
let video = null;
let canvas = null;
let ctx = null;
let isRunning = false;
let trackedObjects = [];
let detections = [];
let isDrawing = false;
let drawStart = null;
let drawEnd = null;
let selectedBox = null;
let lastDetectionTime = 0;
let nextObjectId = 1;

// DOM Elements
const videoEl = document.getElementById('video');
const canvasEl = document.getElementById('canvas');
const statusEl = document.getElementById('status');
const btnStart = document.getElementById('btn-start');
const btnClear = document.getElementById('btn-clear');
const objectsList = document.getElementById('objects-list');
const instructionsEl = document.getElementById('instructions');
const modalEl = document.getElementById('naming-modal');
const nameInput = document.getElementById('object-name');
const btnConfirm = document.getElementById('btn-confirm');
const btnCancel = document.getElementById('btn-cancel');

// Initialize
async function init() {
    video = videoEl;
    canvas = canvasEl;
    ctx = canvas.getContext('2d');
    
    // Set canvas size
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Event listeners
    btnStart.addEventListener('click', toggleTracking);
    btnClear.addEventListener('click', clearAll);
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('dblclick', onDoubleClick);
    
    btnConfirm.addEventListener('click', confirmNaming);
    btnCancel.addEventListener('click', closeModal);
    nameInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') confirmNaming();
    });
    
    // Load model
    statusEl.textContent = 'Loading model...';
    try {
        const modelBuffer = await fetch('yolov8n.onnx').then(r => r.arrayBuffer());
        session = await ort.InferenceSession.create(modelBuffer);
        statusEl.textContent = 'Model loaded';
        console.log('Model loaded successfully');
    } catch (err) {
        statusEl.textContent = 'Model load failed';
        console.error('Failed to load model:', err);
    }
}

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

// Start/Stop tracking
function toggleTracking() {
    if (isRunning) {
        stopTracking();
    } else {
        startTracking();
    }
}

async function startTracking() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 1280, height: 720 }
        });
        video.srcObject = stream;
        await video.play();
        
        video.style.display = 'block';
        isRunning = true;
        btnStart.textContent = '⏹ Stop';
        btnStart.classList.add('active');
        instructionsEl.classList.add('visible');
        
        setTimeout(() => instructionsEl.classList.remove('visible'), 3000);
        
        detectLoop();
    } catch (err) {
        statusEl.textContent = 'Camera error: ' + err.message;
        console.error('Camera error:', err);
    }
}

function stopTracking() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    isRunning = false;
    btnStart.textContent = '▶ Start';
    btnStart.classList.remove('active');
}

// Detection loop
async function detectLoop() {
    if (!isRunning) return;
    
    const now = Date.now();
    if (now - lastDetectionTime >= CONFIG.detectionInterval) {
        await detect();
        lastDetectionTime = now;
    }
    
    draw();
    updateObjectsList();
    requestAnimationFrame(detectLoop);
}

// Run YOLO detection
async function detect() {
    if (!session || !video.videoWidth) return;
    
    // Prepare input
    const [w, h] = [video.videoWidth, video.videoHeight];
    const size = 640;
    
    // Create canvas for model input
    const inputCanvas = document.createElement('canvas');
    inputCanvas.width = size;
    inputCanvas.height = size;
    const inputCtx = inputCanvas.getContext('2d');
    
    // Draw video (mirrored)
    inputCtx.translate(size, 0);
    inputCtx.scale(-1, 1);
    inputCtx.drawImage(video, 0, 0, w, h, 0, 0, size, size);
    
    const inputData = new Float32Array(1 * 3 * size * size);
    const imageData = inputCtx.getImageData(0, 0, size, size);
    
    // Normalize to [0, 1] then to [-1, 1]
    for (let i = 0; i < size * size * 3; i++) {
        inputData[i] = (imageData.data[i] / 255.0) * 2 - 1;
    }
    
    // Run inference
    try {
        const output = await session.run({ images: inputData });
        const outputData = output[Object.keys(output)[0]].data;
        
        // Parse YOLO output
        detections = parseYOLOOutput(outputData, size, w, h);
    } catch (err) {
        console.error('Detection error:', err);
    }
}

// Parse YOLO output to bounding boxes
function parseYOLOOutput(output, inputSize, imgW, imgH) {
    const boxes = [];
    const numClasses = 80;
    const numAnchors = 8400;
    
    // Output shape: [1, 84, 8400] -> [8400, 84]
    // Each anchor: [x, y, w, h, class0, class1, ...]
    
    for (let i = 0; i < numAnchors; i++) {
        const baseIdx = i * (4 + numClasses);
        
        // Get max class score
        let maxScore = 0;
        let maxClass = 0;
        for (let c = 0; c < numClasses; c++) {
            const score = output[baseIdx + 4 + c];
            if (score > maxScore) {
                maxScore = score;
                maxClass = c;
            }
        }
        
        if (maxScore >= CONFIG.confidenceThreshold) {
            // Convert from center format to corner format
            // Note: input is mirrored, so we need to flip x coordinates
            const cx = output[baseIdx] / inputSize * imgW;
            const cy = output[baseIdx + 1] / inputSize * imgH;
            const bw = output[baseIdx + 2] / inputSize * imgW;
            const bh = output[baseIdx + 3] / inputSize * imgH;
            
            // Flip x back (since video is mirrored)
            const x1 = imgW - cx - bw / 2;
            const y1 = cy - bh / 2;
            const x2 = x1 + bw;
            const y2 = y1 + bh;
            
            boxes.push({
                x1: Math.max(0, x1),
                y1: Math.max(0, y1),
                x2: Math.min(imgW, x2),
                y2: Math.min(imgH, y2),
                confidence: maxScore,
                class: maxClass
            });
        }
    }
    
    // Apply NMS
    return nms(boxes, CONFIG.iouThreshold);
}

// Non-Maximum Suppression
function nms(boxes, iouThreshold) {
    if (boxes.length === 0) return [];
    
    boxes.sort((a, b) => b.confidence - a.confidence);
    
    const result = [];
    const suppressed = new Array(boxes.length).fill(false);
    
    for (let i = 0; i < boxes.length; i++) {
        if (suppressed[i]) continue;
        
        result.push(boxes[i]);
        
        for (let j = i + 1; j < boxes.length; j++) {
            if (suppressed[j]) continue;
            if (iou(boxes[i], boxes[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

function iou(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);
    
    if (x2 < x1 || y2 < y1) return 0;
    
    const inter = (x2 - x1) * (y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - inter;
    
    return union > 0 ? inter / union : 0;
}

// Tracking: match detections to tracked objects
function updateTracking() {
    const now = Date.now();
    
    // Filter old trajectory points
    trackedObjects.forEach(obj => {
        obj.trajectory = obj.trajectory.filter(p => now - p.t < CONFIG.trajectoryMaxDuration);
    });
    
    // Match detections to tracked objects
    trackedObjects.forEach(obj => {
        let bestMatch = null;
        let bestIoU = 0;
        
        for (const det of detections) {
            const iouScore = iou(obj.box, det);
            if (iouScore > bestIoU && iouScore > CONFIG.iouThreshold) {
                bestIoU = iouScore;
                bestMatch = det;
            }
        }
        
        if (bestMatch) {
            obj.box = { ...bestMatch };
            obj.lastSeen = now;
            obj.isTracking = true;
            
            // Add trajectory point
            const centerX = (bestMatch.x1 + bestMatch.x2) / 2;
            const centerY = (bestMatch.y1 + bestMatch.y2) / 2;
            
            const lastPoint = obj.trajectory[obj.trajectory.length - 1];
            if (!lastPoint || 
                Math.hypot(centerX - lastPoint.x, centerY - lastPoint.y) > CONFIG.trajectoryMinDistance ||
                now - lastPoint.t > CONFIG.trajectoryMinTime) {
                obj.trajectory.push({ x: centerX, y: centerY, t: now });
            }
        } else {
            obj.isTracking = false;
        }
    });
}

// Draw everything
function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!video.videoWidth) return;
    
    // Calculate scale to fit video to canvas
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    const scale = Math.min(scaleX, scaleY);
    
    const offsetX = (canvas.width - video.videoWidth * scale) / 2;
    const offsetY = (canvas.height - video.videoHeight * scale) / 2;
    
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    
    // Draw detections (if no tracked objects)
    if (trackedObjects.length === 0) {
        detections.forEach(det => {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
            ctx.setLineDash([]);
        });
    }
    
    // Update and draw tracked objects
    updateTracking();
    
    trackedObjects.forEach(obj => {
        const box = obj.box;
        const color = obj.color;
        
        // Draw trajectory
        if (obj.trajectory.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = CONFIG.trajectoryLineWidth;
            ctx.moveTo(obj.trajectory[0].x, obj.trajectory[0].y);
            for (let i = 1; i < obj.trajectory.length; i++) {
                ctx.lineTo(obj.trajectory[i].x, obj.trajectory[i].y);
            }
            ctx.stroke();
        }
        
        // Draw bounding box (only if tracking)
        if (obj.isTracking) {
            ctx.fillStyle = color + '1A'; // hex + alpha
            ctx.fillRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            ctx.strokeStyle = color;
            ctx.lineWidth = CONFIG.boxLineWidth;
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
            
            // Draw label
            const label = `${obj.name} ${Math.round(obj.confidence * 100)}%`;
            const labelWidth = ctx.measureText(label).width + 10;
            ctx.fillStyle = color;
            ctx.fillRect(box.x1, box.y1 - 20, labelWidth, 20);
            ctx.fillStyle = '#000';
            ctx.font = '12px JetBrains Mono';
            ctx.fillText(label, box.x1 + 5, box.y1 - 5);
        }
    });
    
    // Draw selection box
    if (isDrawing && drawStart && drawEnd) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        const x = Math.min(drawStart.x, drawEnd.x);
        const y = Math.min(drawStart.y, drawEnd.y);
        const w = Math.abs(drawEnd.x - drawStart.x);
        const h = Math.abs(drawEnd.y - drawStart.y);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
    }
    
    ctx.restore();
}

// Mouse handlers
function onMouseDown(e) {
    if (!isRunning) return;
    isDrawing = true;
    drawStart = { x: e.clientX, y: e.clientY };
    drawEnd = { x: e.clientX, y: e.clientY };
}

function onMouseMove(e) {
    if (!isDrawing) return;
    drawEnd = { x: e.clientX, y: e.clientY };
}

function onMouseUp(e) {
    if (!isDrawing) return;
    isDrawing = false;
    
    // Get box in video coordinates
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (canvas.width - video.videoWidth * scale) / 2;
    const offsetY = (canvas.height - video.videoHeight * scale) / 2;
    
    const x1 = Math.min(drawStart.x, drawEnd.x);
    const y1 = Math.min(drawStart.y, drawEnd.y);
    const x2 = Math.max(drawStart.x, drawEnd.x);
    const y2 = Math.max(drawStart.y, drawEnd.y);
    
    const box = {
        x1: (x1 - offsetX) / scale,
        y1: (y1 - offsetY) / scale,
        x2: (x2 - offsetX) / scale,
        y2: (y2 - offsetY) / scale
    };
    
    if (box.x2 - box.x1 > 20 && box.y2 - box.y1 > 20) {
        selectedBox = box;
        nameInput.value = `Stone ${nextObjectId++}`;
        modalEl.classList.remove('hidden');
        nameInput.focus();
        nameInput.select();
    }
    
    drawStart = null;
    drawEnd = null;
}

function onDoubleClick(e) {
    // Remove object on double-click
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (canvas.width - video.videoWidth * scale) / 2;
    const offsetY = (canvas.height - video.videoHeight * scale) / 2;
    
    const clickX = (e.clientX - offsetX) / scale;
    const clickY = (e.clientY - offsetY) / scale;
    
    trackedObjects = trackedObjects.filter(obj => {
        const box = obj.box;
        return !(clickX >= box.x1 && clickX <= box.x2 && 
                 clickY >= box.y1 && clickY <= box.y2);
    });
}

function confirmNaming() {
    const name = nameInput.value.trim();
    if (name && selectedBox) {
        const color = COLORS[(nextObjectId - 1) % COLORS.length];
        trackedObjects.push({
            id: nextObjectId,
            name: name,
            color: color,
            box: selectedBox,
            confidence: 1,
            trajectory: [],
            lastSeen: Date.now(),
            isTracking: true
        });
    }
    closeModal();
}

function closeModal() {
    modalEl.classList.add('hidden');
    selectedBox = null;
    nameInput.value = '';
}

function clearAll() {
    trackedObjects = [];
    nextObjectId = 1;
}

function updateObjectsList() {
    objectsList.innerHTML = trackedObjects.map(obj => `
        <div class="object-item">
            <div class="object-color" style="background: ${obj.color}"></div>
            <span class="object-name">${obj.name}</span>
            <span class="object-status ${obj.isTracking ? 'tracking' : 'waiting'}">
                ${obj.isTracking ? '● Tracking' : '○ Waiting'}
            </span>
        </div>
    `).join('');
}

// Start
init();
