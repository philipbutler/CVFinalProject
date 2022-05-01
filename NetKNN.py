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


# load the images, scale down, convert to grey scale and invert the intensities and save to csv
def write_data(train_loader, model, layer_name):
    """
    :param train_loader: npz file, train data and train label
    :param model: keras network model, pre trained model
    :param layer_name: str, the name of which layer to truncate
    :return:
        None
    """
    train_data = train_loader["train_data"]
    train_label = train_loader["train_label"].astype(int)

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(train_data)

    np.savetxt('data.csv', intermediate_output, delimiter=',')
    np.savetxt('label.csv', train_label, delimiter=',')

    print(intermediate_output.shape)
    return


# main function
def main(argv):
    # labels
    name = ['changling', 'phil', 'erica', 'jp']

    # load data
    filepath = 'processedData/'
    train_loader, test_loader = load_data(filepath)

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

    # write the KNN features and create the feature space
    if argv[1] == "KNN":
        model = load_model('CNN.h5')
        layer_name = 'dense_1'
        write_data(train_loader, model, layer_name)

    return


if __name__ == '__main__':
    main(sys.argv)
    # mnist()
