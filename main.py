# ---------- IMPORTS ----------
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from flask import Flask, jsonify
from collections import defaultdict

# ---------- FLASK SERVER ----------
app = Flask(__name__)
latest_detection = {"object": "none", "distance": 0}

@app.route("/result")
def result():
    return jsonify(latest_detection)

@app.route("/")
def home():
    with open("mobile_feedback.html", "r", encoding="utf-8") as f:
        return f.read()

def update_detection(obj, dist):
    """Update the latest detection for Flask endpoint."""
    global latest_detection
    latest_detection = {"object": obj, "distance": dist, "timestamp": time.time()}

# Run Flask server in background
threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
).start()
print("🌐 Flask server started — open http://<your-laptop-IP>:5000/result on phone")

# ---------- SETTINGS ----------
CONFIDENCE_THRESHOLD = 0.5  # minimum detection confidence
SPEAK_DELAY = 3             # delay between voice feedback (handled on mobile)
FOCAL_LENGTH = 2400.0       # from your calibration (adjust if needed)

# Known object widths in centimeters (approximate real-world sizes)
KNOWN_WIDTHS = {
    "person": 40.0, "bicycle": 150.0, "car": 180.0, "motorcycle": 90.0,
    "bus": 250.0, "chair": 45.0, "bottle": 7.0, "dog": 30.0, "cat": 25.0,
    "laptop": 33.0, "cup": 8.0, "backpack": 30.0
}
DEFAULT_KNOWN_WIDTH = 50.0  # fallback width
_smoothed = defaultdict(lambda: None)  # for smoothing distance
ALPHA = 0.4  # smoothing factor

# ---------- LOAD MODEL ----------
print("🔄 Loading SSD MobileNet model...")
detect_fn = tf.saved_model.load('saved_model')
print("✅ Model loaded successfully!")

# ---------- LOAD COCO LABEL MAP ----------
category_index = {}
with open('mscoco_label_map.pbtxt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith("id:"):
            id = int(line.split(": ")[1])
        elif line.startswith("display_name:"):
            name = line.split(": ")[1].replace('"', '')
            category_index[id] = {'id': id, 'name': name}

# ---------- CAMERA SETUP ----------
cap = cv2.VideoCapture(1)  # DroidCam feed
if not cap.isOpened():
    print("❌ Could not connect to mobile camera.")
    exit()
print("🎥 Starting detection... Press 'q' to quit.")

last_spoken_time = 0

# ---------- DETECTION LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received from camera. Retrying...")
        cap = cv2.VideoCapture(1)
        continue

    # Fix orientation
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

    try:
        detections = detect_fn(input_tensor)
    except Exception as e:
        print("⚠️ TensorFlow inference error:", e)
        continue

    num_detections = int(detections.pop('num_detections'))
    detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    height, width, _ = frame.shape
    found_object = False

    for i in range(num_detections):
        score = detections['detection_scores'][i]
        if score > CONFIDENCE_THRESHOLD:
            cls_id = detections['detection_classes'][i]
            box = detections['detection_boxes'][i]
            y1, x1, y2, x2 = box
            (left, top, right, bottom) = (x1 * width, y1 * height, x2 * width, y2 * height)
            label = category_index.get(cls_id, {'name': 'Unknown'})['name']

            # Draw bounding box
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(score * 100)}%", (int(left), int(top) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # ---------- DISTANCE ESTIMATION ----------
            box_width = max(int(right - left), 1)
            known_width = KNOWN_WIDTHS.get(label.lower(), DEFAULT_KNOWN_WIDTH)

            distance_raw = (known_width * FOCAL_LENGTH) / box_width
            prev = _smoothed[label]
            distance = distance_raw if prev is None else ALPHA * distance_raw + (1 - ALPHA) * prev
            _smoothed[label] = distance

            distance = max(10.0, min(distance, 10000.0))  # clamp 10cm–100m

            # ---------- UPDATE SERVER ----------
            update_detection(label, int(distance))

            # Print for debugging
            current_time = time.time()
            if current_time - last_spoken_time > SPEAK_DELAY:
                print(f"🔍 {label} detected ~ {int(distance)} cm ahead")
                last_spoken_time = current_time

            found_object = True

    if not found_object:
        update_detection("none", 0)

    cv2.imshow("Mobile Camera Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Detection stopped.")
