# Changling Li & Phil Butler
# CS 5330
# Final Project
# Convolutional network for recognition

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
import imgProcessing


# class definitions, create the neural network
class FaceNetwork(nn.Module):
    # initialize the network
    # input size (1, 100, 100)
    def __init__(self):
        super().__init__()
        # build layers
        # 20@92*92
        self.conv1 = nn.Conv2d(1, 20, kernel_size=10)
        # maxpooling layer, relu in forward
        # 20@46*46
        self.pool1 = nn.MaxPool2d(2)
        # 40@42*42
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        # dropout layer
        self.dropout = nn.Dropout(p=0.5)
        # maxpooling layer, relu in forward
        # 40@21*21
        self.pool2 = nn.MaxPool2d(2)
        # flatten layer in forward
        self.f1 = nn.Linear(40 * 21 * 21, 100)
        # relu in forward, we have four categories
        self.f2 = nn.Linear(100, 4)
        # log_softmax loss fucntion
        self.f3 = nn.LogSoftmax()

    # computes a forward pass for the network
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(self.dropout(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        return x

# reads train and test data from filepath and returns the arrays
def load_data(filepath):
    train_loader = np.load(filepath + "train_data.npz")
    test_loader = np.load(filepath + "test_data.npz")
    return train_loader, test_loader

# train the network with the input
def train_network(model, epochs, batch, batch_sz, train_loader, test_loader):
    # define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # define loss function
    criterion = nn.CrossEntropyLoss()
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    train_data = train_loader["train_data"]
    train_label = train_loader["train_label"]

    num_samps = train_data.shape[0]

    test_data = torch.from_numpy(test_loader["test_data"].reshape((test_loader["test_data"].shape[0], 1, test_loader["test_data"].shape[1], test_loader["test_data"].shape[2]))).float()
    test_label = torch.from_numpy(test_loader["test_label"])

    for epoch in range(epochs):
        # start training
        correct_count = 0
        total_count = 0
        running_loss = 0.0

        # train the network
        for i in range(batch):
            mini_batch_indices = np.random.choice(num_samps, batch_sz, replace=True)
            mini_batch_true_class = torch.from_numpy(train_label[mini_batch_indices])
            mini_batch = train_data[mini_batch_indices, :, :].reshape((batch_sz, 1, train_data.shape[1], train_data.shape[2]))
            optimizer.zero_grad()
            # print(mini_batch.shape)

            mini_batch = torch.from_numpy(mini_batch)
            mini_batch = mini_batch.float()
            # print(mini_batch.data.size())
            # print(mini_batch.type())
            # forward
            outputs = model(mini_batch)
            loss = criterion(outputs, mini_batch_true_class)
            # calculate the accuracy
            _, prediction = torch.max(outputs, 1)
            total_count += num_samps
            correct_count += (prediction == mini_batch_true_class).sum()
            # backward, optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f'[{epoch + 1}, {i + 1:5d}] train loss: {running_loss / batch_sz:.3f}')
            print(f'[{epoch + 1}, {i + 1:5d}] train accuracy: {correct_count * 1.0 / total_count:.3f}')
            train_loss.append(running_loss / batch_sz)
            train_acc.append(correct_count * 1.0 / total_count)
            running_loss = 0.0

        # test the network
        optimizer.zero_grad()
        correct_count = 0
        total_count = 0
        running_loss_test = 0.0

        outputs = model(test_data)
        loss = criterion(outputs, test_label)
        _, prediction = torch.max(outputs.data, 1)
        total_count += test_data.data.size()[0]
        correct_count += (prediction == test_label).sum()
        running_loss_test += loss.item()

        print(f'[{epoch + 1}, {i + 1:5d}] test loss: {running_loss_test / test_data.data.size()[0]:.3f}')
        print(f'[{epoch + 1}, {i + 1:5d}] test accuracy: {correct_count * 1.0 / total_count:.3f}')
        test_loss.append(running_loss_test / test_data.data.size()[0])
        test_acc.append(correct_count * 1.0 / total_count)
        running_loss_test = 0.0


    return train_loss, train_acc, test_loss, test_acc

def main():

    # labels
    name = ['changling', 'phil', 'erica', 'jp']

    # load data
    filepath = 'processedData/'
    train_loader, test_loader = load_data(filepath)

    # create the network
    network = FaceNetwork()
    epochs = 5
    batch = 5
    batch_sz = 80
    # train the network
    train_loss, train_acc, test_loss, test_acc = train_network(network, epochs, batch, batch_sz, train_loader, test_loader)

    # save the trained model
    torch.save(network.state_dict(), "trained_net.pt")

    # plot the loss and accuracy
    x_train = np.linspace(0, len(train_loss) * 100, len(train_loss))
    x_test = np.linspace(0, len(train_loss) * 100, len(test_loss))
    fig, axs = plt.subplots(2)
    # loss
    axs[0].set_title("Loss vs Batch")
    axs[0].plot(x_train, train_loss, label="Train")
    axs[0].scatter(x_test, test_loss, c='r', label="Test")
    axs[0].set_xlabel("Batch")
    axs[0].set_ylabel("Loss")
    axs[0].legend(loc='upper right')
    # accuracy
    axs[1].set_title("Accuracy vs Batch")
    axs[1].plot(x_train, train_acc, label="Train")
    axs[1].scatter(x_test, test_acc, c='r', label="Test")
    axs[1].set_xlabel("Batch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend(loc='upper right')
    plt.show()

    # Run each data point through network, putting it into the new space
    # Position of labels will still correspond to the position of their data
    # data_in_new_space = []
    # for d in torch.from_numpy(data):
    #
    #     # Put pixel values into interval [0, 1]
    #     d = d / 255
    #
    #     # Add dimension, (28, 28) -> (1, 28, 28), to be accepted by network
    #     d = torch.unsqueeze(d, 0)
    #
    #     output = network(d)
    #     data_in_new_space.append(output)
    #
    # print("First 3 points:")
    # _ = [print(x) for x in data_in_new_space[:3]]

if __name__ == '__main__':
    main()
