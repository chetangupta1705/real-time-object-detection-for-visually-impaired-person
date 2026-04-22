import tensorflow as tf
import cv2

print("🔄 Loading SSD MobileNet model...")
model = tf.saved_model.load('saved_model')
print("✅ Model loaded successfully!")

print("🎥 Checking webcam access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not access the webcam.")
else:
    print("✅ Webcam connected successfully!")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Webcam Test", frame)
        print("📸 Showing webcam feed... Press 'q' to close the window.")
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

print("🧩 Test complete!")
