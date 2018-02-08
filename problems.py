from os.path import exists
from datautils import dataloader, splitXY
from torch.optim import Adam

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

    def train(self, model, loss_fn, hyperparams):
        # retrieve prediction scheme (past and future frames)
        pinds, finds = self.predictscheme()

        # retrieve hyperparameters
        lr     = hyperparams["lr"]
        epochs = hyperparams["epochs"]
        bsize  = hyperparams["bsize"]

        # load problem data from disk
        traindata = dataloader(self.traindirs, nframes=self.horizon(), batch_size=bsize)
        devdata   = dataloader(self.devdirs, nframes=self.horizon(), batch_size=bsize)

        # choose Adam as the optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        # setup directory for TensorBoard summaries
        basename = "runs/{},{},lr={}".format(model.name(), self.colorspace(), lr)
        dirname, attempt = basename, 0
        while exists(dirname):
            attempt += 1
            dirname = basename + " (" + str(attempt) + ")"

        writer = SummaryWriter(dirname)

        for epoch in range(epochs):
            for (iteration, batch) in enumerate(traindata):
                # features and targets
                X, Y = splitXY(batch, pinds, finds)

                # start clean
                optimizer.zero_grad()

                # propagate forward
                Yhat = model(X)

                # compute loss
                yhat = Yhat.view(Yhat.size(0), -1)
                y    = Y.view(Y.size(0), -1)
                loss = loss_fn(yhat, y)

                # update TensorBoard summary
                writer.add_scalar("loss/train", loss, iteration)
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

        writer.close()

        return loss.data.numpy()[0]
