# Erica Shepherd, 
# CS 5330
# Final Project
# Live Face Recognition

# import statements 
import cv2 as cv
import faceDetection as fd
import faceRecognition as fr
import imgProcessing as ip

# main function
def main():
    capdev = cv.VideoCapture(0, cv.CAP_DSHOW)

    if not capdev.isOpened():
        print("Error: unable to open camera")
        exit()

    # load cascades
    faceCascade, eyeCascade, smileCascade = fd.loadCascades()

    # labels
    name = ['changling', 'phil', 'erica', 'jp']

    # create and read in trained models
    LBPHrecognizer, eigenfaces, fisherfaces = fr.loadModels("opencvModels/")

    # controls pause
    pause = False

    # keeps track of mode
    mode = 0

    # video stream
    while(1):
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
                                            minNeighbors=5, minSize=(30,30))
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

        # draws features on frame if user presses d
        elif key == ord('d'):
            mode = 1
        # LBPH face recognition
        elif key == ord('l'):
            mode = 2
        # eigenfaces recognition
        elif key == ord('e'):
            mode = 3
        # fisherfaces recognition
        elif key == ord('f'):
            mode = 4
        

        # modes
        # face features detection
        if mode == 1:
            fd.drawFeatures(frame, faces, (255, 0, 0))
            fd.drawFeatures(frame, eyes, (0, 255, 0))
            fd.drawFeatures(frame, smile, (0, 0, 255))  
        # LBPH recognition
        elif mode == 2:
            fd.drawFeatures(frame, faces, (255, 0, 0))
            for (x, y, w, h) in faces:
                id, confidence = LBPHrecognizer.predict(grayFrame[y:y+h, x:x+w])
                cv.putText(frame, str(name[id]), (x+5, y-5), fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1, color=(255,255,255), thickness=2)
        # eigenfaces recognition
        elif mode == 3:
            fd.drawFeatures(frame, faces, (255, 0, 0))
            for (x, y, w, h) in faces:
                tempFrame = ip.img_process(frame[y:y+h, x:x+w], (100,100))
                id, confidence = eigenfaces.predict(tempFrame)
                cv.putText(frame, str(name[id]), (x+5, y-5), fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1, color=(255,255,255), thickness=2)
        # fisherfaces recognition
        elif mode == 4:
            fd.drawFeatures(frame, faces, (255, 0, 0))
            for (x, y, w, h) in faces:
                tempFrame = ip.img_process(frame[y:y+h, x:x+w], (100,100))
                id, confidence = fisherfaces.predict(tempFrame)
                cv.putText(frame, str(name[id]), (x+5, y-5), fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1, color=(255,255,255), thickness=2)
         

        cv.imshow("Video", frame) 

    # end video stream
    capdev.release() 
    cv.destroyAllWindows()

    return

# runs code only if in file
if __name__ == "__main__":
    main()