import torch
from torch.nn import Module, MSELoss, L1Loss, BCELoss

class TVLoss(Module):
    def __init__(self, cspace):
        super(TVLoss, self).__init__()

        if cspace == "BW":
            self.loss = BCELoss()
        elif cspace == "GRAY":
            self.loss = L1Loss()
        else:
            self.loss = MSELoss()

    def forward(self, yhat, y):
        bsize, chan, height, width = y.size()
        errors = []
        for h in range(height-1):
            dy    = y[:,:,h+1,:] - y[:,:,h,:]
            dyhat = yhat[:,:,h+1,:] - yhat[:,:,h,:]
            error = torch.norm(dy - dyhat, 1)
            errors.append(error)

        E = sum(errors) / height / bsize

        return self.loss(yhat, y) + 0.01*E
