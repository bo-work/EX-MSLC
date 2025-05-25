import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class Encoder(nn.Module):

    # def __init__(self, layers_dims=(60, 40, 20), input_shape=[80]):
    def __init__(self, layers_dims=(60, 40, 20)):
        super(Encoder, self).__init__()

        self.layers_dims = layers_dims
        # self.input_shape = input_shape
        self.layers = []
        self.net = None

        self.outputs = None

        sinWeights = []
        sinCoeffs = []

        for i in range(1, len(self.layers_dims)):
            self.layers.append(nn.Linear(self.layers_dims[i-1], self.layers_dims[i], bias=True))

            t = torch.randn(self.layers_dims[i - 1], 1, requires_grad=True)
            sinWeights.append(nn.Parameter(Variable(t, requires_grad=True)))

            t = torch.randn(1, 1, requires_grad=True)
            sinCoeffs.append(nn.Parameter(Variable(t, requires_grad=True)))

        self.sinWeights = sinWeights
        self.sinCoeffs = sinCoeffs

        self.net = nn.Sequential(*self.layers)
        self.sinWeights = nn.ParameterList(self.sinWeights)
        self.sinCoeffs = nn.ParameterList(self.sinCoeffs)

        # # add
        # if torch.cuda.is_available():
        #     self.sinWeights = self.sinWeights.cuda()
        #     self.sinCoeffs = self.sinCoeffs.cuda()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, save=True):

        for i in range(len(self.net) - 1):
            x = self.net[i](x)
            x = self.sinCoeffs[i+1] * torch.sin(2 * np.pi * x * self.sinWeights[i+1].view(1, -1))
        x = self.net[-1](x)
        x = x.view(-1, self.layers_dims[-1])

        if save:
            self.outputs = x

        return x
