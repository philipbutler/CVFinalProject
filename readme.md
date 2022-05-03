# Final Project: Face Detection, Recognition, and Filter Application
### Changling Li, Philip Butler, Jiapei Li, Erica Shepherd

### [Presentation & Demo](link), [Full Paper](link)

## Machine and Software Specifications
This code was tested on the following OS:
- macOS Big Sur v.11.4
- macOS macOS Monterey v.12.3.1
- Windows 11
and in general should run on any machine with python 3.9 and required modules

## How to run
#### 1. Install Required Python Modules
Assuming you have python3.9 and pip installed, before you potentially alter an existing python environment you have, you'll probably want to [create a new virtual environment](https://docs.python.org/3/library/venv.html).
Once it's activated, install the necessary modules in requirements.txt by running
```sh
pip3 install -r requirements.txt
```

#### 2. Consider Command Line Arguments
##### liveRecognition.py - Run program as intended
On macOS, in general, users should be able to run liveRecognition.py with no command line arguments.
```sh
python3 liveRecognition.py
```
On Windows, if this doesn't work, you'll want to try
```sh
python3 liveRecognition.py windows
```
If you've installed the EpocCam driver on your computer ([which allows you to use your iPhone camera input](https://www.elgato.com/en/epoccam)),
you may need to specify that you still actually want to use your webcam.
```sh
python3 liveRecognition.py webcam
```
##### NetKNN.py


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



