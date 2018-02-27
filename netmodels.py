import torch
import numpy as np
from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import ReLU, Tanh, Sigmoid
from torch.nn.functional import sigmoid

def classname(model):
    return model.__class__.__name__

class CodeRegNet(Module):
    """
    An enconder followed by a regressor.

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

class TorricelliNet(Module):
    """
    Args:
        pastlen: number of past frames
        futurelen: number of future frames
        cspace: color space (BW or RGB)
    """
    def __init__(self, pastlen, futurelen, cspace):
        super(TorricelliNet, self).__init__()

        # problem dimensions
        P = pastlen
        F = futurelen
        C = 3 if cspace == "RGB" else 1

        self.velocity = Sequential(
            Conv2d(P*C, 10*P*C, kernel_size=5, padding=2),
            BatchNorm2d(10*P*C), Tanh(),
            Conv2d(10*P*C, P*C, kernel_size=5, padding=2)
        )

        self.acceleration = Sequential(
            Conv2d(P*C, 10*P*C, kernel_size=5, padding=2),
            BatchNorm2d(10*P*C), Tanh(),
            Conv2d(10*P*C, P*C, kernel_size=5, padding=2)
        )

        self.predict = Sequential(
            Conv2d(P*C, F*C, kernel_size=1),
            BatchNorm2d(F*C), Sigmoid()
        )

        self.cspace = cspace

    def forward(self, x):
        v = self.velocity(x)
        a = self.acceleration(x)
        y = self.predict(x + v + a/2)
        return y
