from torch.nn import Module, Sequential
from torch.nn import Conv2d, ReLU
from torch.nn.functional import sigmoid

def classname(model):
    return model.__class__.__name__

class CodecNet(Module):
    """
    A codec (i.e. code-decode) module.

    Args:
        inchan: number of input channels
        outchan: number of output channels
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
