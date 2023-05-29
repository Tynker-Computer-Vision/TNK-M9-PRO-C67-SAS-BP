Detect All Possible Objects
============================
In this activity, you will learn to detect all the possible objects from a image.

<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10509013/aa2.gif" width = "480" height = "320">


Follow the given steps to complete this activity:

1. Detect the objects

* Open the main.py file.

* Add name of the detected object using `labels[classIds[i]]`.

    `cv2.putText(image, labels[classIds[i]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)`

* Save and run the code to check the output.

