# Final Project: Face Detection, Recognition, and Filter Application
### Changling Li, Philip Butler, Jiapei Li, Erica Shepherd

### Description
This project will detect faces and face features such as eyes & mouth. It will identify faces, and apply different filters and/or objects relative to the face dependent on whose face it is (also different actions like smiling may trigger swapping filters). The program will be implemented in Python. We will be experimenting with different algorithms for face detection/recognition and comparing the results. 

### [Presentation & Demo](https://youtu.be/lEFt5HL0Y1Q), [Full Paper](https://drive.google.com/file/d/1QdAcXYKpIu3cmwrZCdxToBrYpJiFBC9E/view?usp=sharing)

## Machine and Software Specifications
This code was tested on the following OS:
- macOS Big Sur v.11.4
- macOS Monterey v.12.3.1
- Windows 11

and in general should run on any machine with python 3.9 and required modules

## How to run
#### 1. Install Required Python Modules
Assuming you have python3.9 and pip installed, before you potentially alter an existing python environment you have, you'll probably want to [create a new virtual environment](https://docs.python.org/3/library/venv.html).
Once it's activated, install the necessary modules in requirements.txt by running
```sh
pip install -r requirements.txt
```

#### 2. Consider Command Line Arguments
##### liveRecognition.py - Run program as intended
On macOS, in general, users should be able to run liveRecognition.py with no command line arguments.
```sh
python liveRecognition.py
```
On Windows, if this doesn't work, you'll want to try
```sh
python liveRecognition.py windows
```
If you've installed the EpocCam driver on your computer ([which allows you to use your iPhone camera input](https://www.elgato.com/en/epoccam)),
you may need to specify that you still actually want to use your webcam.
```sh
python liveRecognition.py webcam
```


## Controls
| Key | Action|
| ------ | ------ |
| p | pause |
| b | toggle bounding box |
| n | toggle displaying name |
| f | toggle filters |
| 0 | turn everything off |
| 1 | draw all boxes |
| 2-6 | recognition model mode |
| q | quit |



