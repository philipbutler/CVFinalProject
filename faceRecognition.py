# Erica Shepherd
# CS 5330
# Final Project
# LBPH Face Recognition

# import statements 
import cv2 as cv
import numpy as np

# reads train and test data from filepath and returns the arrays
def loadData(filepath):
    trainData = np.load(filepath + "train_data.npz")
    testData = np.load(filepath + "test_data.npz")
    return trainData, testData

# tests model on test data
def testModel(recognizer, testData):
    totalCount = len(testData["test_data"])
    correct = 0

    # runs through test data and checks predictions with ground truth
    for i in range(totalCount):
        prediction, confidence = recognizer.predict(testData["test_data"][i])
        groundTruth = testData["test_label"][i]

        if (prediction == groundTruth):
            correct += 1

    # returns accuracy of predictions
    accuracy = (correct / totalCount) * 100
    return accuracy

# main function
def main():
    # loads in training and test data
    trainData, testData = loadData("processedData/")

    # create and train network
    recognizer = cv.face.LBPHFaceRecognizer_create() # if not installed already, run the command (python -m pip install --user opencv-contrib-python)
    recognizer.train(trainData["train_data"], trainData["train_label"])

    # saves network
    recognizer.write("LBPHmodel.xml")

    # tests and prints accuracy
    accuracy = testModel(recognizer, testData)
    print("{:.2f}%".format(accuracy))

    return

# runs code only if in file
if __name__ == "__main__":
    main()