# Erica Shepherd
# CS 5330
# Final Project
# Haar Cascade Face detection

# import statements
import cv2 as cv
import os

# draws rectangles in given color around detected features in given frame
def drawFeatures(frame, features, color):
    for (x, y, w, h) in features:
        frame = cv.rectangle(img=frame, pt1=(x,y), pt2=(x+w, y+h), 
                                color=color, thickness=2)

# returns detected face features in face region of frame
def findFeatures(grayFrame, faces, featureCascade, scaleFactor=1.3, 
                                    minNeighbors=5, minSize=(0, 0)):
    featuresFinal = []

    for (x, y, w, h) in faces:
        # checks for features only in face region
        regionOfInterest = grayFrame[y:y+h, x:x+w]
        features = featureCascade.detectMultiScale(regionOfInterest, scaleFactor=scaleFactor,
                                                minNeighbors=minNeighbors, minSize=minSize)

        # re-adjusts values back into frame values
        for (fx, fy, fw, fh) in features:
            feature = (fx+x, fy+y, fw, fh)
            featuresFinal.append(feature)
        
    return featuresFinal

# creates folder if path does not exist
def checkDirectory(path):
    if os.path.exists(path)==False:
        os.makedirs(path)
        print("Creating", path)

# main function
def main():
    capdev = cv.VideoCapture(0, cv.CAP_DSHOW)

    if not capdev.isOpened():
        print("Error: unable to open camera")
        exit()

    # load pre-trained classifier from OpenCV directory (https://github.com/opencv/opencv/tree/master/data/haarcascades)
    faceCascade = cv.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
    eyeCascade = cv.CascadeClassifier("cascades/haarcascade_eye_tree_eyeglasses.xml")
    smileCascade = cv.CascadeClassifier("cascades/haarcascade_smile.xml")

    # used for saving images
    userID = ""
    imageCount = 1

    # video stream
    while(1):
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
                                            minNeighbors=5, minSize=(30,30))
        eyes = findFeatures(grayFrame, faces, eyeCascade, scaleFactor=1.3, 
                                            minNeighbors=3, minSize=(3, 3))
        smile = findFeatures(grayFrame, faces, smileCascade, scaleFactor=2, 
                                            minNeighbors=30, minSize=(50, 50))

        # draws features on frame
        drawFeatures(frame, faces, (255, 0, 0))
        drawFeatures(frame, eyes, (0, 255, 0))
        drawFeatures(frame, smile, (0, 0, 255))

        cv.imshow("Video", frame)

        # key commands
        key = cv.waitKey(1)

        # saves face region of original image if user presses s
        if key == ord('s'):
            # obtains userID if none already assigned
            if userID == "":
                userID = input("Please enter your name: ")
                print("Now saving images for", userID)

            # creates path directory if not already made
            dirPath = "dataset"
            checkDirectory(dirPath)
            dirPath += "/" + userID
            checkDirectory(dirPath)
            
            # saves grayscale images of all faces
            for (x, y, w, h) in faces:
                path = dirPath + "/" + str(imageCount) + ".jpg"
                cv.imwrite(path, grayFrame[y:y+h, x:x+w])
                imageCount += 1

        # quits if user presses q
        elif key == ord('q'):
            break        

    # end video stream
    capdev.release() 
    cv.destroyAllWindows()

    return

# runs code only if in file
if __name__ == "__main__":
    main()