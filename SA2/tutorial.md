Object Detection and Labeling
==============================

In this activity, you will learn to change the color of the higlighting box in the video.

<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10495375/person_ball.gif" width = "480" height = "320">

Follow the given steps to complete this activity:

1. Detect sports ball

* Open the main.py file.

* Check if the `classIds` of the labels is equal to `sports ball`.

    `if labels[classIds[i]] == "sports ball":`

* Start a infinite while loop to read the frames of video continously.

    `if i % 2 == 0:`

        color = (0, 255, 0)

    `else:`

        color = (255, 0, 0)
`
* Get the labels for the classIds and store them in variable label .

    `label = labels[classIds[i]]`

* Using the `putText()` function add the labels.

    `cv2.putText(image, label, (x, y - 8),font, 0.7, color, 2)`

* Save and run the code to check the output.

2. Detect person

* Open the main.py file.

* Check if the `classIds` of the labels is equal to person.

    `if labels[classIds[i]] == "person":`

* Start a infinite while loop to read the frames of video continously.

    `if i % 2 == 0:`

        `color = (0, 255, 0)`

    `else:`

        `color = (255, 0, 0)`

* Get the labels for the `classIds` and store them in variable `label` .

    `label = labels[classIds[i]]`

* Using the `putText()` function add the labels.

    `cv2.putText(image, label, (x, y - 8),font, 0.7, color, 2)`

* Save and run the code to check the output.
