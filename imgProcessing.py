# Changling Li
# CS 5330
# Final Project
# Load image, pre-process image, create data set, labels and save to npz files

import cv2
import os
import sys
import numpy as np


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
    return grey_img


# load the images, scale down to a given resolution, and store the data and label
def load_img(filepath, labels, size):
    """
    :param filepath: string, directory where the images are located
    :param labels: list, labels of the images
    :param size: tuple, desired input size to the network
    :return:
        data_set: numpy array, all the image data
        label_set: numpy array, all the labels
    """
    all_path = os.listdir(filepath)
    if ".DS_Store" in all_path:
        all_path.remove(".DS_Store")

    all_data = []
    all_labels = []

    # load each img
    for path in all_path:
        current_path = filepath + path
        all_img = os.listdir(current_path)
        if ".DS_Store" in all_img:
            all_img.remove(".DS_Store")
        # append the labels
        for idx in range(len(labels)):
            if labels[idx] in current_path:
                all_labels.extend([idx for i in range(len(all_img))])
        # append the img data
        for img in all_img:
            current_img_path = current_path + '/' + img
            print(current_img_path)
            current_img = cv2.imread(current_img_path)
            processed_img = img_process(current_img, size)
            all_data.append(processed_img)

    if len(all_data) == len(all_labels):
        print("all data has the same length as all labels")

    data_set = np.array(all_data)
    label_set = np.array(all_labels)

    return data_set, label_set


# shuffle the data and create the training set, training label, test set and test label
def train_test_split(data_set, label_set, ratio=0.2):
    """
    :param data_set: numpy array, all the image data
    :param label_set: numpy array, all the corresponding labels
    :param ratio: float, the percentage of data as test set
    :return:
        train_set: numpy array, training data
        train_label: numpy array, training data corresponding labels
        test_set: numpy array, testing data
        test_label: numpy array, testing data corresponding labels
    """
    shuffler = np.random.permutation(len(label_set))
    shuffled_data = data_set[shuffler]
    shuffled_label = label_set[shuffler]

    n_train_samps = int((1 - ratio) * len(label_set))
    train_set = shuffled_data[:n_train_samps, :]
    test_set = shuffled_data[n_train_samps:, :]

    train_label = shuffled_label[:n_train_samps]
    test_label = shuffled_label[n_train_samps:]

    return train_set, train_label, test_set, test_label


# save the data and labels to npz file
def save_data(train_data, train_label, test_data, test_label):
    np.savez('processedData/train_data.npz', train_data=train_data, train_label=train_label)
    np.savez('processedData/test_data.npz', test_data=test_data, test_label=test_label)
    return


# main function
def main(argv):
    labels = ['changling', 'phil', 'erica', 'Jiapei']
    file_path = 'dataset/'
    data_set, label_set = load_img(file_path, labels, (100, 100))
    train_set, train_label, test_set, test_label = train_test_split(data_set, label_set)
    print(train_set.shape)
    print(train_label.shape)
    print(test_set.shape)
    print(test_label.shape)
    save_data(train_set, train_label, test_set, test_label)


if __name__ == '__main__':
    main(sys.argv)
