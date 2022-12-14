# rock-paper-spock
This toy application uses mediapipe, cv2.ml, and streamlit to play Rock Paper Scissors Lizard Spock.  

Setup:
```bash
python -m venv venv
. venv/bin/activate
python -m pip install -r requirement.in
```

or 
```
setup.bat
```

Run:
```bash
./rpsls.bat
```

Currently, rock-paper-spock uses open cv's knn algorithm with k=5.
The dataset includes 10 examples of each of the five gestures.
For each gesture the angle between the joints (starting with the wrist) on each finger is saved (5*3).
It is possible that including the angle between the fingers might improve results (paper vs spock).
You can use ./train.bat to add additional examples.

!["example game"](example.png "example")
_Figure 1.  Spock vaporizes rock_

Note: you may have to increase CAMERA in rpsls.py to get your camera to work


Derived from [Rock paper scissors using #mediapipe and #streamlit python](https://www.youtube.com/watch?v=ee29JMl41Mc)
