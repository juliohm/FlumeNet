from problems import VideoGenProblem
from netmodels import CodecNet
from torch.nn import MSELoss, BCELoss, L1Loss
from plotting import movie

# define the problem
problem = VideoGenProblem("data/bw", ["data/bw/Run 1"], cspace="BW")

# define the network model
model = CodecNet(problem.pastlen()*problem.channels(),
                 problem.futurelen()*problem.channels(), cspace="BW")

# define the loss criterion
loss_fn = MSELoss() if problem.colorspace() == "RGB" else BCELoss()

hyperparams = {
    "lr": 0.01,
    "epochs": 3,
    "bsize": 64
}

# solve the problem with the model
solution, losses = problem.solve(model, loss_fn, hyperparams)

# generate a video with the trained network
movie(solution, "data/bw/Run 1")
