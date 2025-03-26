import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("models/yolo.weights", "models/yolo.cfg")
layer_names = net.getUnconnectedOutLayersNames()

# Load video
cap = cv2.VideoCapture("data/object_detection.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Draw detections on frame
    for detection in outputs:
        confidence = detection[5]
        if confidence > 0.5:
            x, y, w, h = detection[0:4]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
