# TensorBoard imports
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

class TBoardDebugger():
    """
    TensorBoard debugger.

    Args:
        module: a PyTorch module to debug
        linds: indices of layers to debug
        cinds: channels of layers to debug
    """
    def __init__(self, module, linds, cinds):
        self.module = module
        self.linds = linds
        self.cinds = cinds
        self.writer = SummaryWriter("runs/debug")
        self.iter = 0

    def consume(self, x):
        y = x.clone()
        for (l,layer) in enumerate(self.module):
            y = layer(y)
            if l in self.linds:
                for c in self.cinds:
                    grid = vutils.make_grid(y.data[:,c:c+1,:,:], normalize=True, scale_each=True)
                    grid = grid.cpu().numpy().transpose([1, 2, 0])
                    self.writer.add_image("activations/layer-{}-channel-{}".format(l,c), grid, self.iter)
