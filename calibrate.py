import cv2
import tensorflow as tf
import numpy as np

detect_fn = tf.saved_model.load('saved_model')
cap = cv2.VideoCapture(1)

KNOWN_DISTANCE = 50.0   # in cm
KNOWN_WIDTH = 14.0      # in cm

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    height, width, _ = frame.shape

    for i in range(num_detections):
        score = detections['detection_scores'][i]
        if score > 0.5:
            box = detections['detection_boxes'][i]
            y1, x1, y2, x2 = box
            left, right = int(x1 * width), int(x2 * width)
            perceived_width = right - left

            if perceived_width > 0:
                focal_length = (perceived_width * KNOWN_DISTANCE) / KNOWN_WIDTH
                print(f"📏 Focal Length ≈ {focal_length:.2f} pixels")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Calibration Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
