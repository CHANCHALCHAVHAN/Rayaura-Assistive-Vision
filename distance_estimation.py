import cv2
import numpy as np
import time

# Load YOLO Model
net = cv2.dnn.readNet("models/yolo.weights", "models/yolo.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Load video
cap = cv2.VideoCapture("data/driving_test.mp4")

frame_rate = 30  # FPS
collision_threshold = 2.0  # Time in seconds to collision

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Distance Calculation (Dummy Example)
    for detection in outputs:
        confidence = detection[5]
        if confidence > 0.5:
            x, y, w, h = detection[0:4]
            distance = 500 / h  # Approximate formula for distance
            time_to_collision = distance / 30  # Assuming 30 km/h speed

            if time_to_collision < collision_threshold:
                cv2.putText(frame, "⚠️ COLLISION WARNING!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Distance Estimation", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
