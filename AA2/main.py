import numpy as np
import cv2
confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

video = cv2.VideoCapture("bb2.mp4")

state = "play"

while True:
    if (state == "play"):
        check, image = video.read()

        if (check):
            image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

            dimensions = image.shape[:2]
            H, W = dimensions
            blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416))
            yoloNetwork.setInput(blob)

            layerName = yoloNetwork.getUnconnectedOutLayersNames()
            layerOutputs = yoloNetwork.forward(layerName)

            boxes = []
            confidences = []
            classIds = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]

                    if confidence > confidenceThreshold:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY,  width, height) = box.astype('int')
                        x = int(centerX - (width/2))
                        y = int(centerY - (height/2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIds.append(classId)

            indexes = cv2.dnn.NMSBoxes(
                boxes, confidences, confidenceThreshold, NMSThreshold)

            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(boxes)):
                if i in indexes:
                    # Make yolo detect all possible objects
                    if labels[classIds[i]] == "sports ball":
                        x, y, w, h = boxes[i]
                        color = (255, 0, 0)
                        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                        # Add name of the detected object

            cv2.imshow('Image', image)
            cv2.waitKey(1)

    key = cv2.waitKey(1)
    if key == 32:
        print("Stopped")
        break
    if key == 112:
        state = "pause"
    if key == 108:
        state = "play"
    if key == 114:
        video = cv2.VideoCapture("bb2.mp4")
