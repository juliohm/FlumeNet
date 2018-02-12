import numpy as np
from os import listdir, path
from imageio import imread

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class FlumeData(Dataset):
    """
    Data from a single run of the flume experiment.

    Args:
        rundir:  directory containing run images
        nframes: number of frames per training example
    """
    def __init__(self, rundir, nframes=2, augment=True):
        self.rundir = rundir
        self.nframes = nframes
        self.augment = augment
        self.filenames = sorted(listdir(rundir))

    def __len__(self):
        return len(self.filenames) - self.nframes + 1

    def __getitem__(self, ind):
        imgs = []
        for i in range(ind, ind + self.nframes):
            img = np.asarray(imread(path.join(self.rundir, self.filenames[i])))
            img = img if img.ndim == 3 else img[...,np.newaxis]
            img = np.float32(np.transpose(img, [2, 0, 1]) / 255)
            imgs.append(img)

        if self.augment and np.random.rand() > 0.5:
            # flip images horizontaly at random
            imgs = [img[:,:,::-1].copy() for img in imgs]

        return imgs

class MixedFlumeData(Dataset):
    """
    Data from multiple runs of the flume experiment.

    Args:
        rundirs: directories containing run images
        nframes: number of frames per training example
    """
    def __init__(self, rundirs, nframes=2, augment=True):
        self.nframes = nframes
        self.datasets = [FlumeData(rundir, nframes, augment) for rundir in rundirs]

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, ind):
        N = 0
        for d in self.datasets:
            n = len(d)
            if ind < N + n:
                return d[ind - N]
            N += n

def loadimages(dirs, nframes=2, augment=True, **kwargs):
    """
    Create Torch DataLoader for datasets in separate directories.

    Args:
        dirs: a root directory (e.g. "data/bw") containing subdirectories
              for each run of the experiment, or a list of strings
              (e.g. ["data/bw/Run 1", "data/bw/Run 2"]) with specific runs.
        nframes: number of frames per training example
        kwargs: keyword arguments for Torch DataLoader (e.g. batch_size)
    """
    rundirs = sorted([path.join(dirs, d) for d in listdir(dirs)]) if type(dirs) is str else dirs
    dataset = MixedFlumeData(rundirs, nframes, augment)
    return DataLoader(dataset, **kwargs)

def splitXY(batch, pinds, finds):
    """
    Splits batch of examples into X representing frames in the past
    and Y representing frames in the future.

    Args:
        pinds: indices of frames in the past to stack as features (e.g. range(T))
        finds: indices of frames in the future to stack as labels (e.g. [T])
    """
    X = torch.cat([batch[t] for t in pinds], 1)
    Y = torch.cat([batch[t] for t in finds], 1)
    X = Variable(X, requires_grad=False)
    Y = Variable(Y, requires_grad=False)

    return X, Y
