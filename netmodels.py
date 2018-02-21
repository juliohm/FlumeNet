import torch
import numpy as np
from torch.nn import Module, Sequential
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d
from torch.nn.functional import sigmoid

def classname(model):
    return model.__class__.__name__

class CodecNet(Module):
    """
    A codec (i.e. code-decode) module.

    Args:
        inchan: number of input channels
        outchan: number of output channels
        cspace: color space (BW or RGB)
    """
    def __init__(self, inchan, outchan, cspace):
        super(CodecNet, self).__init__()
        self.model = Sequential(
            Conv2d(inchan, 32, kernel_size=3, padding=1), ReLU(),
            Conv2d(32, outchan, kernel_size=3, padding=1)
        )
        self.cspace = cspace

    def forward(self, x):
        y = self.model(x)
        return sigmoid(y) if self.cspace == "BW" else y

class FlumeNet(Module):
    """
    An enconder followed by a convolutional regressor.

    Args:
        pastlen: number of past frames
        futurelen: number of future frames
        cspace: color space (BW or RGB)
    """
    def __init__(self, pastlen, futurelen, cspace):
        super(FlumeNet, self).__init__()

        # problem dimensions
        P = pastlen
        F = futurelen
        C = 3 if cspace == "RGB" else 1

        self.encoder = Sequential(
            Conv2d(C, 64, kernel_size=3, padding=1), ReLU(),
            Conv2d(64, 128, kernel_size=5, padding=2), ReLU(),
            Conv2d(128, 10, kernel_size=5, padding=2)
        )

        self.regressor = Sequential(
            Conv2d(P*10, F*C, kernel_size=3, padding=1)
        )

        # save attributes
        self.cspace = cspace
        self.P = P
        self.F = F
        self.C = C

    def forward(self, x):
        frames = [x[:,i*self.C:(i+1)*self.C,:,:] for i in range(self.P)]
        encodings = [self.encoder(frame) for frame in frames] # encode
        encodings = torch.cat(encodings, 1) # stack encodings

        y = self.regressor(encodings)

        return sigmoid(y) if self.cspace == "BW" else y
