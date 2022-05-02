# Erica Shepherd
# CS 5330
# Final Project
# LBPH/Eigenfaces/Fisherfaces Face Recognition

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

# reads and returns saved opencv models
def loadModels(path):
    LBPHrecognizer = cv.face.LBPHFaceRecognizer_create()
    LBPHrecognizer.read(path + "LBPHmodel.xml")

    eigenfaces = cv.face.EigenFaceRecognizer_create()
    eigenfaces.read(path + "eigenfacesModel.xml")

    fisherfaces = cv.face.FisherFaceRecognizer_create()
    fisherfaces.read(path + "fisherfacesModel.xml")

    return LBPHrecognizer, eigenfaces, fisherfaces

# main function
def main():
    # loads in training and test data
    trainData, testData = loadData("processedData/")

    # create and train recognizers
    # LBPH recognizer
    LBPHrecognizer = cv.face.LBPHFaceRecognizer_create() # if not installed already, run the command (python -m pip install --user opencv-contrib-python)
    LBPHrecognizer.train(trainData["train_data"], trainData["train_label"])
    # eigenfaces
    eigenfaces = cv.face.EigenFaceRecognizer_create()
    eigenfaces.train(trainData["train_data"], trainData["train_label"])
    # fisherfaces
    fisherfaces = cv.face.FisherFaceRecognizer_create()
    fisherfaces.train(trainData["train_data"], trainData["train_label"])

    # saves models
    path = "opencvModels/"
    LBPHrecognizer.write(path + "LBPHmodel.xml")
    eigenfaces.write(path + "eigenfacesModel.xml")
    fisherfaces.write(path + "fisherfacesModel.xml")

    # tests and prints accuracies on test data
    LBPHaccuracy = testModel(LBPHrecognizer, testData)
    eigenfacesAccuracy = testModel(eigenfaces, testData)
    fisherfacesAccuracy = testModel(fisherfaces, testData)
    print("LBPH algorithm: {:.2f}%".format(LBPHaccuracy))
    print("Eigenfaces: {:.2f}%".format(eigenfacesAccuracy))
    print("Fisherfaces: {:.2f}%".format(fisherfacesAccuracy))

    return

# runs code only if in file
if __name__ == "__main__":
    main()