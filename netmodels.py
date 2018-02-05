from torch.nn import Module, ModuleList
from torch.nn import Conv2d, MaxPool2d, Upsample

class CodecNet(Module):
    """
    A codec (i.e. code-decode) module.

    Args:
        cod_chans: number of filters in encoding layers
        dec_chans: number of filters in decoding layers
    """
    def __init__(self, cod_chans=[1,16,32], dec_chans=None):
        if dec_chans is None:
            dec_chans = cod_chans[::-1]

        assert len(cod_chans) > 1, "too few enconding layers"
        assert len(dec_chans) > 1, "too few decoding layers"

        super(CodecNet, self).__init__()

        self.layers = ModuleList()
        for l in range(len(cod_chans)-1):
            self.layers.append(Conv2d(cod_chans[l], cod_chans[l+1], 3, padding=1))
            self.layers.append(MaxPool2d(2, stride=2))
        for l in range(len(dec_chans)-1):
            self.layers.append(Conv2d(dec_chans[l], dec_chans[l+1], 3, padding=1))
            self.layers.append(Upsample(scale_factor=2))

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
