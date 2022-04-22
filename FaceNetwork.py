import os
import torch
from torch import nn
import torch.nn.functional as F
import imgProcessing

# Neural Network Class. Specifies an architecture for this model.
class FaceNetwork(nn.Module):
    def __init__(self):
        super(FaceNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # Computes a forward pass for the network
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():

    # Currently the network from P5.
    network = FaceNetwork()
    data, labels = imgProcessing.load_img(os.getcwd() + '/dataset/', ['phil'], (28, 28))

    # Run each data point through network, putting it into the new space
    # Position of labels will still correspond to the position of their data
    data_in_new_space = []
    for d in torch.from_numpy(data):

        # Put pixel values into interval [0, 1]
        d = d / 255

        # Add dimension, (28, 28) -> (1, 28, 28), to be accepted by network
        d = torch.unsqueeze(d, 0)

        output = network(d)
        data_in_new_space.append(output)

    print("First 3 points:")
    _ = [print(x) for x in data_in_new_space[:3]]

if __name__ == '__main__':
    main()
