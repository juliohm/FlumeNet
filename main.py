from problems import VideoGenProblem
from netmodels import CodecNet
from torch.nn import MSELoss, BCEWithLogitsLoss

problem = VideoGenProblem("data/rgb", "data/rgb", cspace="RGB")

model = CodecNet(problem.pastlen()*problem.channels(),
                 problem.futurelen()*problem.channels())

loss_fn = MSELoss() if problem.colorspace() == "RGB" else BCEWithLogitsLoss()

hyperparams = {
    "lr": 0.001,
    "epochs": 1,
    "bsize": 64
}

loss = problem.train(model, loss_fn, hyperparams)
