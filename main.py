from problems import VideoGenProblem
from netmodels import CodecNet
from torch.nn import MSELoss, BCEWithLogitsLoss

problem = VideoGenProblem("data/rgb", "data/rgb", cspace="RGB")

m = problem.pastlen()
n = problem.futurelen()
c = problem.channels()

model = CodecNet(m*c, n*c)

loss_fn = MSELoss() if problem.colorspace() == "RGB" else BCEWithLogitsLoss()

hyperparams = {
    "lr": 0.001,
    "epochs": 1,
    "bsize": 64
}

loss = problem.train(model, loss_fn, hyperparams)
