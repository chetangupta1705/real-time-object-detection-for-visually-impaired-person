import cv2

cap = cv2.VideoCapture(1)  # try 0 or 1 depending on your setup
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("DroidCam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
