Add Play and Pause
====================
In this activity, you will learn to add the play and pause functionality.

<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10495500/play_pause.gif" width = "480" height = "320">

Follow the given steps to complete this activity:

1. Add play state

* Open the main.py file.

* Declare state varible and set it to play.

    `state = "play"`

* check if `state` is `play` inside the while loop.

    `if (state == "play"):`

* Shift all the code inside this if condition.

* Close the window when the space bar key is pressed.

    `key = cv2.waitKey(1) if key == 32:`

    `print("Stopped")`

    `break`


* Change the state to pause when p key is pressed.

    `if key == 112:` # p key
    
        `state = "pause"`


* Change the state to play when l key is pressed.
 
    `if key == 108:` # l key
    
        `state = "play"`

* Save and run the code to check the output.
