Read the Input Video
=====================

In this activity, you will learn to read the frames of a video using cv2 library.

<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10495566/readingvideo.gif" width = "480" height = "320">


Follow the given steps to complete this activity:

1. Read the Video

* Open the main.py file.

* Declare a variable `video`.

* Using the `cv2.VideoCapture()` function read the video by passing the video as parameter.

    `video = cv2.VideoCapture("bb2.mp4")`

* Start a infinite while loop to read the frames of video continously.

    `while True:`

* Declare a varible `readVideo` and read the first frame using `video.read()` function.

    `readVideo = video.read()`

* Get the first frame using `readVideo[0]` and set it as a value of check variable.

    `check = readVideo[0]`

* Check if the first frame is present using if condition.

    `if (check):`

* Read the second frame of the video and set it to the image variable.

    `image = readVideo[1]`

* Resize the image using `cv2.resize()` function.

    `image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)`

* Save and run the code to check the output.






