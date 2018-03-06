import torch
import numpy as np
from torch.nn import Module, Sequential, ModuleList
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d
from torch.nn import RNN, GRU, ConvTranspose2d
from torch.nn import ReLU, Sigmoid

def classname(model):
    return model.__class__.__name__

class Flatten(Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class SliceNet(Module):
    """
    A model based on slices of the frames.

    Args:
        pastlen: number of past frames
        futurelen: number of future frames
        cspace: color space (BW or RGB)
        nslice: number of slices to predict
    """
    def __init__(self, pastlen, futurelen, cspace, nslice=30):
        super(SliceNet, self).__init__()

        # problem dimensions
        P = pastlen
        F = futurelen
        C = 3 if cspace == "RGB" else 1

        self.fwdtime = ModuleList([GRU(input_size=C*100, hidden_size=C*100) for s in range(nslice)])
        self.fillgap = ModuleList([GRU(input_size=C*100, hidden_size=C*100) for s in range(nslice)])

        # save attributes
        self.P = P
        self.F = F
        self.C = C
        self.nslice = nslice

    def forward(self, x):
        # retrieve frames
        frames = [x[:,i*self.C:(i+1)*self.C,:,:] for i in range(self.P)]

        # spacing between horizontal slices
        dx = 150 // self.nslice

        # run prediction forward in time for each slice
        tslices = []
        for s in range(self.nslice):
            slices = [frame[:,:,s*dx,:].contiguous() for frame in frames]
            flattened = [s.view(s.size(0), -1) for s in slices]

            # stack slices from different frames for recurrence in time
            augmented = [s[np.newaxis,...] for s in flattened]
            stacked = torch.cat(augmented, 0)

            # forward pass in time
            _, hidden = self.fwdtime[s](stacked)

            # save time prediction
            tslices.append(hidden)


        # fill in the gaps between the horizontal slices
        allslices = []
        for s in range(self.nslice):
            x = tslices[s]
            allslices.append(x)
            for _ in range(dx-1):
                __, x = self.fillgap[s](x)
                allslices.append(x)

        # reshape slices to image format
        imgslices = [s.view(s.size(1), self.C, 1, 100) for s in allslices]
        image = torch.cat(imgslices, 2)

        return image

class FlumeNet(Module):
    """
    An enconder followed by a recurrent module followed by a decoder.

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
            Conv2d(C, 64, kernel_size=3, padding=1),
            BatchNorm2d(64), ReLU(),
            MaxPool2d(2),
            Conv2d(64, 128, kernel_size=5, stride=5),
            BatchNorm2d(128), ReLU(),
            MaxPool2d(3),
            Flatten()
        )

        self.recurrence = Sequential(
            RNN(input_size=5*3*128, hidden_size=20*20*C, num_layers=1)
        )

        self.decoder = Sequential(
            ConvTranspose2d(F*C, F*C, kernel_size=3, stride=2),
            BatchNorm2d(F*C), ReLU(),
            ConvTranspose2d(F*C, F*C, kernel_size=3, stride=2),
            BatchNorm2d(F*C), ReLU(),
            ConvTranspose2d(F*C, F*C, kernel_size=3, stride=2),
            Flatten(), Sigmoid()
        )

        # save attributes
        self.P = P
        self.F = F
        self.C = C

    def forward(self, x):
        # encode frames
        frames = [x[:,i*self.C:(i+1)*self.C,:,:] for i in range(self.P)]
        encodings = [self.encoder(frame) for frame in frames]

        # stack sequence of encodings for RNN
        encodings = [enc[np.newaxis,...] for enc in encodings]
        encodings = torch.cat(encodings, 0)

        # pass through recurrent layers
        encodings = self.recurrence(encodings)

        # reshape hidden state to image format
        hidden = encodings[1][0,:,:]
        nbatch, nfeat = hidden.size()
        w = int(np.sqrt(nfeat // self.C))
        hidden = hidden.view(nbatch, self.C, w, w)

        # decode to original size
        output = self.decoder(hidden)
        output = output[:,:150*100*self.C].contiguous()

        # reshape and return
        image = output.view(output.size(0), self.C, 150, 100)

        return image

class TorricelliNet(Module):
    """
    A simple model inspired by Torricelli's equations of motion.

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
            BatchNorm2d(10*P*C), ReLU(),
            Conv2d(10*P*C, P*C, kernel_size=5, padding=2)
        )

        self.acceleration = Sequential(
            Conv2d(P*C, 10*P*C, kernel_size=5, padding=2),
            BatchNorm2d(10*P*C), ReLU(),
            Conv2d(10*P*C, P*C, kernel_size=5, padding=2)
        )

        self.predict = Sequential(
            Conv2d(P*C, F*C, kernel_size=1),
            BatchNorm2d(F*C), Sigmoid()
        )

    def forward(self, x):
        v = self.velocity(x)
        a = self.acceleration(x)
        y = self.predict(x + v + a/2)
        return y
