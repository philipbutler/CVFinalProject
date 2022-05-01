# Jiapei Li
# CS 5330
# Final Project
# Live implementation of filters

# import statements
from statistics import mode
import cv2 as cv
import sys
import FaceNetwork
from enum import Enum
from filters import *
import faceDetection as fd

class Mode(Enum):
    NORMAL = 1
    FILTER = 2


mode = Mode.NORMAL
counter = 0

# main function
def main(argv):
    print(argv)
    if len(argv) > 1 and argv[1] == 'webcam':
        capdev = cv.VideoCapture(1)
    else:
        capdev = cv.VideoCapture(0)

    if not capdev.isOpened():
        print("Error: unable to open camera")
        exit()

    # load pre-trained classifier from OpenCV directory (https://github.com/opencv/opencv/tree/master/data/haarcascades)
    faceCascade = cv.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
    eyeCascade = cv.CascadeClassifier("cascades/haarcascade_eye_tree_eyeglasses.xml")
    smileCascade = cv.CascadeClassifier("cascades/haarcascade_smile.xml")

    # reference global variable
    global mode
    global counter

    # load filter
    filters = loadFilters()
    print(len(filters))
    cv.imshow("1", filters[0].filter)
    cv.imshow("2", filters[1].filter)

    # video stream, quits if user presses q
    key = cv.waitKey(1)
    while key != ord('q'):
        # captures frame
        ret, frame = capdev.read()

        # ret checks for correct frame read
        if ret is not True:
            print("Error: frame not read correctly")
            exit()    

        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # finds features // note: scale factors, min neighbors, and min size may need
        #                // adjustment to optimize detection
        faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.3,
                                             minNeighbors=5, minSize=(30, 30))
        eyes = fd.findFeatures(grayFrame, faces, eyeCascade, scaleFactor=1.3,
                            minNeighbors=3, minSize=(3, 3))
        smile = fd.findFeatures(grayFrame, faces, smileCascade, scaleFactor=2,
                             minNeighbors=30, minSize=(50, 50))

        # key commands
        key = cv.waitKey(1)

        if key == ord('n'):
            mode = Mode.NORMAL
        if key == ord('f'):
            mode = Mode.FILTER
        if key == ord('u'):
            counter = counter + 1
            mode = Mode.FILTER

        # draws features on frame
        if len(faces) > 0 and mode == Mode.NORMAL:
            fd.drawFeatures(frame, faces, (255, 0, 0))
            fd.drawFeatures(frame, eyes, (0, 255, 0))
            fd.drawFeatures(frame, smile, (0, 0, 255))

        # switch to filter mode
        if len(faces) > 0 and mode == Mode.FILTER:
            applyFilter(frame, faces, filters, counter)

        cv.imshow("Video", frame)

    # end video stream
    capdev.release()
    cv.destroyAllWindows()

    return


# runs code only if in file
if __name__ == "__main__":
    main(sys.argv)
