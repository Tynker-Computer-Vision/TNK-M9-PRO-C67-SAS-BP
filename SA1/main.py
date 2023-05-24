import numpy as np
import cv2

confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Read the input video file


# Define infinite while loop

    # Read the first frame of the video
   

    # Write condition to check the first frame is exist or not
    

        # Display image
        
