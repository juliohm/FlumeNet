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
    def __init__(self, inchan, outchan, cspace="BW"):
        super(CodecNet, self).__init__()
        self.model = Sequential(
            Conv2d(inchan, 32, kernel_size=3, padding=1), ReLU(),
            Conv2d(32, outchan, kernel_size=3, padding=1)
        )
        self.cspace = cspace

    def forward(self, x):
        if self.cspace == "BW":
            y = self.model(x)
            return sigmoid(y)
        else:
            return self.model(x)

class FlumeNet(Module):
    """
    An enconder followed by recurrent module.

    Args:
        cspace: color space (BW or RGB)
    """
    def __init__(self, cspace="BW"):
        super(FlumeNet, self).__init__()

        self.channels = 3 if cspace == "RGB" else 1

        self.encoder = Sequential(
            Conv2d(self.channels, 32, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(2),
            Conv2d(64, 128, kernel_size=3, padding=1), ReLU(),
            MaxPool2d(2),
            Conv2d(128, 256, kernel_size=2, stride=2), ReLU(),
            MaxPool2d(2)
        )

        self.fcs = Sequential(
            Linear(9216, 10000), ReLU(),
            Linear(10000, 15000), ReLU()
        )

    def forward(self, x):
        nframes = int(x.shape[1] / self.channels)
        frames = [x[:,i*self.channels:(i+1)*self.channels,:,:] for i in range(nframes)]

        encs = [self.encoder(frame) for frame in frames] # encode
        encs = [enc.view(enc.size(0), -1) for enc in encs] # flatten

        # stack features from all time steps
        encs = torch.cat(encs, 1)

        # forward into fully connected layers
        encs = self.fcs(encs)

        # reshape to image shape (i.e. unflatten)
        encs = encs.view(encs.size(0), 150, 100)

        return encs
