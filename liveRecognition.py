# Erica Shepherd, Changling Li, Phil Butler, Jiapei Li
# CS 5330
# Final Project
# Live Face Recognition

# import statements 
import cv2 as cv
import numpy as np
import faceDetection as fd
import faceRecognition as fr
import imgProcessing as ip
import NetKNN as net
from keras.models import load_model
from keras.models import Model

import filters
import markers

# main function
def main():
    capdev = cv.VideoCapture(0)

    if not capdev.isOpened():
        print("Error: unable to open camera")
        exit()

    # load cascades
    faceCascade, eyeCascade, smileCascade = fd.loadCascades()

    # load the pretrained network and KNN space
    CNN_model = load_model('CNN.h5')
    train_data_path = 'data.csv'
    train_label_path = 'label.csv'
    layer_name = 'dense_1'
    data_list, label_list = net.load_csv(train_data_path, train_label_path)

    # labels
    name = ['changling', 'phil', 'erica', 'jp']

    # create and read in trained models
    LBPHrecognizer, eigenfaces, fisherfaces = fr.loadModels("opencvModels/")

    # keeps track of pause, recognition, filter modes
    pause, draw_box, filter_mode, display_name = False, False, False, True
    recognition_mode = 0

    # load name to filter map
    name2label = filters.loadPersonNameToLabel()

    # load filters
    filterMap = filters.loadFilters()

    # load background gifs into their respective filters
    gifIdx = 0
    markers.loadGifsToMap(filterMap)

    # Necessary for finding background markers
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

    # video stream
    while True:
        # captures frame if stream not paused
        if not pause:
            ret, frame = capdev.read()

            # ret checks for correct frame read
            if ret is not True:
                print("Error: frame not read correctly")
                exit()

        grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # finds features
        faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.3,
                                             minNeighbors=5, minSize=(30, 30))
        eyes = fd.findFeatures(grayFrame, faces, eyeCascade, scaleFactor=1.3,
                               minNeighbors=3, minSize=(3, 3))
        smile = fd.findFeatures(grayFrame, faces, smileCascade, scaleFactor=2,
                                minNeighbors=30, minSize=(50, 50))

        # key commands
        key = cv.waitKey(1)

        # pauses/unpauses video if user presses p
        if key == ord('p'):
            pause = not pause
        # quits if user presses q
        elif key == ord('q'):
            break

        # reset everything (turn off drawing, recognition, & filters, turn on displaying name)
        if key == ord('0'):
            recognition_mode = 0
            pause, draw_box, filter_mode, display_name = False, False, False, True
        # toggle displaying name if in a recognition mode
        elif key == ord('n'):
            display_name = not display_name
        # toggle drawing blue face box
        elif key == ord('b'):
            draw_box = not draw_box
        # draws features on frame if user presses d
        elif key == ord('1'):
            recognition_mode = 1
        # LBPH face recognition mode
        elif key == ord('2'):
            recognition_mode = 2
        # eigenfaces recognition mode
        elif key == ord('3'):
            recognition_mode = 3
        # fisherfaces recognition mode
        elif key == ord('4'):
            recognition_mode = 4
        elif key == ord('5'):
            recognition_mode = 5
        # KNN recognition
        elif key == ord('6'):
            recognition_mode = 6

        # toggle filters
        elif key == ord('f'):
            filter_mode = not filter_mode
            display_name = not filter_mode

        # modes
        if draw_box:
            fd.drawFeatures(frame, faces, (255, 0, 0))

        # face features detection
        if recognition_mode == 1:
            fd.drawFeatures(frame, faces, (255, 0, 0))
            fd.drawFeatures(frame, eyes, (0, 255, 0))
            fd.drawFeatures(frame, smile, (0, 0, 255))

        elif recognition_mode > 1:
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # LBPH recognition
                if recognition_mode == 2:
                    id, confidence = LBPHrecognizer.predict(grayFrame[y:y + h, x:x + w])
                # eigenfaces recognition
                elif recognition_mode == 3:
                    tempFrame = ip.img_process(frame[y:y + h, x:x + w], (100, 100))
                    id, confidence = eigenfaces.predict(tempFrame)
                # fisherfaces recognition
                elif recognition_mode == 4:
                    tempFrame = ip.img_process(frame[y:y + h, x:x + w], (100, 100))
                    id, confidence = fisherfaces.predict(tempFrame)
                # CNN face recognition
                elif recognition_mode == 5:
                    tempFrame = ip.img_process(frame[y:y + h, x:x + w], (101, 101))
                    input_img = tempFrame.reshape((1, 101, 101, 1))
                    id = CNN_model.predict(input_img).argmax(axis=-1)[0]
                # KNN face recognition
                elif recognition_mode == 6:
                    tempFrame = ip.img_process(frame[y:y + h, x:x + w], (101, 101))
                    input_img = tempFrame.reshape((1, 101, 101, 1))
                    input_features = net.feature_calculate(input_img, CNN_model, layer_name)
                    input_features = input_features.tolist()[0]
                    current_error = net.SSD_list(input_features, data_list)
                    id = net.KNN(current_error, label_list, 5)
                if display_name:
                    cv.putText(frame, str(name[id]), (x + 5, y - 5), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=1, color=(255, 255, 255), thickness=2)

                # Select the filter based on the identified person
                filter_label = name2label[name[id]]
                gif = filterMap[filter_label].gif

                if len(faces) > 0 and filter_mode:
                    filters.applyFilter(frame, faces, filterMap[filter_label])
                    gifIdx = (gifIdx + 1) % len(gif)
                    frame = markers.detectAndShowMarkers(frame, dictionary, gif[gifIdx]).astype(np.uint8)

        cv.imshow("Video", frame)

    # end video stream
    capdev.release()
    cv.destroyAllWindows()

    return


# runs code only if in file
if __name__ == "__main__":
    main()
