from ultralytics import YOLO
import numpy as np
import cv2
import math
from sort import *  # noqa: F403

cap = cv2.VideoCapture('cars.mp4') #for video file

model = YOLO('yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

mask = cv2.imread('car_mask.png')

# Create SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [423, 297, 673, 297]
totalCounts = []
while True:
    sucess, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detection = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #confidence value
            conf = math.ceil((box.conf[0]*100))/100            
            
            #class name
            classIndex = int(box.cls[0])
            if classIndex < len(classNames):
                current_class = classNames[classIndex]
            else:
                pass

            if current_class == 'car' and conf > 0.5:
                current_array = np.array([[x1, y1, x2, y2, conf]])
                detection = np.vstack((detection, current_array))

    # Update tracker
    resultsTracker = tracker.update(detection)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id  = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1

        cv2.putText(img, f'{id}', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 3, (0, 255, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[3] + 15:
            if id not in totalCounts:
                totalCounts.append(id)

            cv2.line(img, (cx, cy), (cx, cy), (255, 0, 0), 1)
            cv2.putText(img, f' Count : {len(totalCounts)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
