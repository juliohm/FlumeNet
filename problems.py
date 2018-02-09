from os.path import exists
from datautils import loadimages, splitXY
from torch.optim import Adam
from tqdm import tqdm

# TensorBoard imports
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

class VideoGenProblem(object):
    def __init__(self, traindirs, devdirs, cspace="BW", pinds=[1,2,3], finds=[4]):
        self.traindirs = traindirs
        self.devdirs = devdirs
        self.cspace = cspace
        self.pinds = pinds
        self.finds = finds

    def colorspace(self):
        return self.cspace.upper()

    def predictscheme(self):
        # convert 1-based to 0-based indices
        ps = [p-1 for p in self.pinds]
        fs = [f-1 for f in self.finds]
        return ps, fs

    def horizon(self):
        indmax = max(max(self.pinds), max(self.finds))
        indmin = min(min(self.pinds), min(self.finds))
        return indmax - indmin + 1

    def pastlen(self):
        return len(self.pinds)

    def futurelen(self):
        return len(self.finds)

    def channels(self):
        return 3 if self.cspace.upper() == "RGB" else 1

    def solve(self, model, loss_fn, hyperparams):
        # retrieve prediction scheme (past and future frames)
        pinds, finds = self.predictscheme()

        # retrieve hyperparameters
        lr     = hyperparams["lr"]
        epochs = hyperparams["epochs"]
        bsize  = hyperparams["bsize"]

        # load problem data from disk
        traindata = loadimages(self.traindirs, nframes=self.horizon(), batch_size=bsize)
        devdata   = loadimages(self.devdirs, nframes=self.horizon(), batch_size=bsize)

        # cycle the dev set with an iterator
        deviter = iter(devdata)

        # choose Adam as the optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        # setup directory for TensorBoard summaries
        basename = "runs/{},{},lr={}".format(model.name(), self.colorspace(), lr)
        dirname, attempt = basename, 0
        while exists(dirname):
            attempt += 1
            dirname = basename + " (" + str(attempt) + ")"

        writer = SummaryWriter(dirname)

        # save hyperparameters for the run
        with open(dirname+"/hyperparams.txt","w") as h:
            for (pname, pval) in hyperparams.items():
                h.write("{}: {}\n".format(pname,pval))

        progress = tqdm(total=epochs*len(traindata))

        for epoch in range(epochs):
            for (iteration, batch) in enumerate(traindata):
                # features and targets
                X, Y = splitXY(batch, pinds, finds)

                # start clean
                optimizer.zero_grad()

                # propagate forward
                Yhat = model(X)

                # compute loss
                y    = Y.view(Y.size(0), -1)
                yhat = Yhat.view(Yhat.size(0), -1)
                loss = loss_fn(yhat, y)

                # do the same for dev set
                devbatch = next(deviter, None)
                if devbatch is None:
                    deviter = iter(devdata)
                    devbatch = next(deviter)
                Xdev, Ydev = splitXY(devbatch, pinds, finds)
                Yhatdev = model(Xdev)
                ydev = Ydev.view(Ydev.size(0), -1)
                yhatdev = Yhatdev.view(Yhatdev.size(0), -1)
                lossdev = loss_fn(yhatdev, ydev)

                # update TensorBoard summary
                writer.add_scalar("loss/train", loss, iteration)
                writer.add_scalar("loss/dev", lossdev, iteration)
                if iteration % 10 == 0:
                    truegrid = vutils.make_grid(Y.data, normalize=True, scale_each=True)
                    predgrid = vutils.make_grid(Yhat.data, normalize=True, scale_each=True)
                    truegrid = truegrid.numpy().transpose([1, 2, 0])
                    predgrid = predgrid.numpy().transpose([1, 2, 0])
                    writer.add_image("images/actual", truegrid, iteration)
                    writer.add_image("images/prediction", predgrid, iteration)
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), iteration, bins="doane")

                # compute gradients
                loss.backward()

                # update parameters
                optimizer.step()

                # update progress bar
                lossval = loss.data.numpy()[0]
                lossdevval = lossdev.data.numpy()[0]
                progress.set_postfix(loss="{:05.3f}".format(lossval), lossdev="{:05.3f}".format(lossdevval))
                progress.update()

        writer.close()

        return lossval, lossdevval
