import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module): # Implementation of the Bottleneck Composite Layer
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate # 4*k ( nb of channels )
        self.bn1 = nn.BatchNorm2d(nChannels) # (batchnorm performed on l*k channels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False) # (1x1 conv)
        self.bn2 = nn.BatchNorm2d(interChannels) # ( Batchnorm performed on 4*k channels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False) # (3x3 conv)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x))) # calculating (1x1 conv) after passing activation function through batchnorm result
        out = self.conv2(F.relu(self.bn2(out))) # calculating (3x3 conv) after passing activation function through batchnorm result
        out = torch.cat((x, out), 1) # concatenating k & l*k
        return out

class SingleLayer(nn.Module): # Composite layer w/out bottleneck
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels) # (batchnorm performed on l*k channels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False) # (1x1 conv)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x))) # calculating (1x1 conv) after passing activation function through batchnorm result
        out = torch.cat((x, out), 1) # concatenating k & l*k
        return out 

class Transition(nn.Module): # TRansition layer between each dense block
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels) # (batchnorm performed on l*k channels output of dense block)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False) # (1x1 conv)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x))) # calculating (1x1 conv) after passing activation function through batchnorm result
        out = F.avg_pool2d(out, 2) # Pooling of result of 1x1 conv
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        # reduction is Compression in paper but not sure what it is.
        super(DenseNet, self).__init__()

        # dense connected layers in each denseblock
        nDenseBlocks = (depth-4) // 3 # not sure to understand
        if bottleneck:
            nDenseBlocks //= 2 # not sure to understand

        # channels before entering the first Dense-Block
        nChannels = 2*growthRate # initial convolution layer comprises 2k convolutions

        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False) ## RGB input (image) applied with conv2d of size (3x3), input convnet

        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck) # Dense Block 1 and transition
        nChannels += nDenseBlocks*growthRate # number of channels at the output of the dense block
        nOutChannels = int(math.floor(nChannels*reduction)) # number of channel at the output of the transition
        self.trans1 = Transition(nChannels, nOutChannels) # Transition layer of the output of the dense block

        nChannels = nOutChannels # input dimension of next block is output dimension of previous block
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck) # Dense Block 2 and transition
        nChannels += nDenseBlocks*growthRate # number of channels at the output of the dense block 2
        nOutChannels = int(math.floor(nChannels*reduction)) # number of channel at the output of the transition 2
        self.trans2 = Transition(nChannels, nOutChannels) # Transition layer of the output of the dense block 2

        nChannels = nOutChannels # input dimension of next block is output dimension of previous block
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck) # Dense Block 3
        nChannels += nDenseBlocks*growthRate # number of channels at the output of the dense block 3

        self.bn1 = nn.BatchNorm2d(nChannels) # batchnorm on output of the dense block 3
        self.fc = nn.Linear(nChannels, nClasses) # neural network with input nChannels ( dim of ouput of block3) and output
                                                    # is the number of classes we want to predict

        # Initializing random parameters for all the parameters of the model.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate)) # adding nDenseBlocks Composite Bottleneck layers
            else:
                layers.append(SingleLayer(nChannels, growthRate)) # adding nDenseBlocks Composite layers
            nChannels += growthRate  # for each number of dense block, incrementing the number of channel by the growthrate
        return nn.Sequential(*layers) # returning all layers of the Denseblock as Sequential layers ( connected )

    def forward(self, x):
        out = self.conv1(x) # Going through input of the convnet
        out = self.trans1(self.dense1(out)) # Going through first dense block
        out = self.trans2(self.dense2(out)) # Going through seconde dense block
        out = self.dense3(out) # Going through third dense block
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8)) # Batch norm followed by activation function and then pooling function
                                                                    # then Flattening to fit Linear model input
        out = F.log_softmax(self.fc(out))  # linear neural network & prediction output made with softmax activation
        return out
