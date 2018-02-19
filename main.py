from problems import VideoGenProblem
from netmodels import CodecNet
from torch.nn import MSELoss, BCELoss, L1Loss

problem = VideoGenProblem("data/bw", ["data/bw/Run 3"], cspace="BW")

model = CodecNet(problem.pastlen()*problem.channels(),
                 problem.futurelen()*problem.channels(), cspace="BW")

loss_fn = MSELoss() if problem.colorspace() == "RGB" else BCELoss()

hyperparams = {
    "lr": 0.01,
    "epochs": 3,
    "bsize": 64
}

solution, losses = problem.solve(model, loss_fn, hyperparams)
