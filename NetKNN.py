# Changling Li
# CS 5330
# Final Project
# Convolutional network for recognition

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.utils import to_categorical
from keras.models import Model
import sys
import cv2
import csv
from keras.models import load_model


def load_data(filepath):
    """
    :param filepath: str, file path of the dataset npz files
    :return:
        train_loader: npz, train data and label
        test_loader: npz, test data and label
    """
    train_loader = np.load(filepath + "train_data.npz")
    test_loader = np.load(filepath + "test_data.npz")
    return train_loader, test_loader


# CNN network class
class CNN():
    def __init__(self):
        self.network = self.build_network()

    def build_network(self):
        """
        :return: keras model
        """
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(101, 101, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(4))
        model.summary()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def train_network(self, train_loader, test_loader, batch_size=80, epochs=10):
        """
        :param train_loader: npz, train data and label
        :param test_loader: npz, test data and label
        :param batch_size: int, batch size for training
        :param epochs: int, number of epochs
        :return:
            history: tensor, history of both training and testing accuracy
        """
        train_data = train_loader["train_data"]
        train_label = train_loader["train_label"]
        test_data = test_loader["test_data"]
        test_label = test_loader["test_label"]

        history = self.network.fit(train_data, train_label, batch_size=batch_size, epochs=epochs,
                                   validation_data=(test_data, test_label))

        return history

    def save_model(self):
        self.network.save("CNN.h5")


# process the image, scale down to a given size, convert to grey scale
def img_process(img, size):
    """
    :param img: mat, input image
    :param size: tuple, desired input size to the network
    :return:
        processed_img: mat, img after processed
    """
    resized = cv2.resize(img, size)
    grey_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm_img = cv2.normalize(grey_img, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return grey_img

# calculates features
def feature_calculate(train_data, model, layer_name):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(train_data)
    return intermediate_output


# load the images, scale down, convert to grey scale and invert the intensities and save to csv
def write_data(train_loader, model, layer_name):
    """
    :param train_loader: npz file, train data and train label
    :param model: keras network model, pre trained model
    :param layer_name: str, the name of which layer to truncate
    :return:
        None
    """
    train_label = train_loader["train_label"].astype(int)
    train_data = train_loader["train_data"]

    intermediate_output = feature_calculate(train_data, model, layer_name)

    np.savetxt('data.csv', intermediate_output, delimiter=',')
    np.savetxt('label.csv', train_label, delimiter=',')

    print(intermediate_output.shape)
    return


# load the csv data file and reshape, load the label file
def load_csv(datafile, labelfile):
    """
    :param datafile: str, data csv file name
    :param labelfile: str, label csv file name
    :return:
        data_list: list, data list
        label_list: list, label list
    """
    # read the data file and reshape each rule to 28x28 and save to a list
    data_list = []
    with open(datafile, "r") as f1:
        data_reader = csv.reader(f1, quoting=csv.QUOTE_NONNUMERIC)
        for row in data_reader:
            data_list.append(row)
    # read the label file and save to a list
    label_list = []
    with open(labelfile, "r") as f2:
        label_reader = csv.reader(f2, quoting=csv.QUOTE_NONNUMERIC)
        for row in label_reader:
            label_list.append(int(row[0]))

    return data_list, label_list


# compute the sum squared distance
def SSD(list_a, list_b):
    """
    :param list_a: list
    :param list_b: list
    :return:
        error: int, ssd error
    """
    error = 0
    for i in range(len(list_a)):
        error += (list_a[i] - list_b[i]) ** 2
    return error


# compute all the SSD and save to a list
def SSD_list(list_a, all_list):
    """
    :param list_a: list
    :param all_list: list
    :return:
        error_list: list, all ssd list
    """
    error_list = []
    for list in all_list:
        error = SSD(list_a, list)
        error_list.append(error)
    return error_list


# K-NN classifier
def KNN(error_list, labels, K, drop_first=False):
    """
    :param error_list: list
    :param labels: list
    :param K: int, top K neighbors
    :param drop_first:
    :return:
        int, the index for the label
    """
    error_list, labels = (list(t) for t in zip(*sorted(zip(error_list, labels))))
    if drop_first:
        new_labels = labels[1:K + 1]
    else:
        new_labels = labels[0:K]
    return max(set(new_labels), key=new_labels.count)


# main function
def main(argv):
    # labels
    name = ['changling', 'phil', 'erica', 'jp']

    # load data
    filepath = 'processedData/'
    train_loader, test_loader = load_data(filepath)

    if len(argv) < 2:
        print("User message: <argument>")

    if argv[1] == "train":
        # create the network
        network = CNN()
        epochs = 10

        # train the network
        history = network.train_network(train_loader, test_loader)

        network.save_model()

        # plot the training and testing accuracy
        plt.plot(history.history['accuracy'], label='train_accuracy')
        plt.plot(history.history['val_accuracy'], label='test_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
    if argv[1] == "CNN_predict":
        model = load_model('CNN.h5')
        img = cv2.imread('dataset/changling/99.jpg')
        processed_img = img_process(img, (101, 101))
        input_img = processed_img.reshape((1, 101, 101, 1))
        prediction = model.predict(input_img).argmax(axis=-1)[0]
        print("the prediction is ", prediction)

    # write the KNN features and create the feature space
    if argv[1] == "KNN":
        model = load_model('CNN.h5')
        layer_name = 'dense_1'
        write_data(train_loader, model, layer_name)

    # evaluate the performance of the KNN classification
    if argv[1] == "KNN_evaluation":
        train_data_path = 'data.csv'
        train_label_path = 'label.csv'
        data_list, label_list = load_csv(train_data_path, train_label_path)
        test_label = test_loader['test_label']
        test_data = test_loader['test_data']

        model = load_model('CNN.h5')
        layer_name = 'dense_1'

        # img = cv2.imread('dataset/changling/99.jpg')
        # processed_img = img_process(img, (101, 101))
        # input_img = processed_img.reshape((1, 101, 101, 1))
        # input_features = feature_calculate(input_img, model, layer_name)
        # input_features = input_features.tolist()[0]
        # current_error = SSD_list(input_features, data_list)
        # prediction = KNN(current_error, label_list, 5)
        # print(prediction)

        test_features = feature_calculate(test_data, model, layer_name)
        test_features = test_features.tolist()
        predictions = []
        for data in test_features:
            current_error = SSD_list(data, data_list)
            prediction = KNN(current_error, label_list, 5)
            predictions.append(prediction)
        print("the length of the predication list is ", len(predictions))
        predictions = np.array(predictions)

        num_accurate_prediction = np.sum(predictions == test_label)

        print("KNN accuracy on test set is ", num_accurate_prediction/test_label.shape[0])
    return

# runs code only if in file
if __name__ == '__main__':
    main(sys.argv)
    # mnist()
