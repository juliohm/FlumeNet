from problems import VideoGenProblem
from netmodels import TorricelliNet, SliceNet
from torch.nn import MSELoss, BCELoss, L1Loss
from plotting import movie, diffplot
from losses import TVLoss

# color space
cspace = "RGB"

# data directory
if cspace == "BW":
    prefix = "data/bw/"
elif cspace == "GRAY":
    prefix = "data/gray/"
elif cspace == "FLOW":
    prefix = "data/flow/"
else:
    prefix = "data/rgb/"

# datasets
traindirs = ["run1", "run2.1", "run2.2", "run3.2", "run4",
             "run5", "run6.1", "run6.2", "run7.1", "run7.2"]
devdirs   = ["run3.1"]

traindirs = [prefix+tdir for tdir in traindirs]
devdirs   = [prefix+ddir for ddir in devdirs]

# define the problem
problem = VideoGenProblem(traindirs, devdirs, cspace=cspace, pinds=[1,2,3], finds=[4])

# define the network model
model = TorricelliNet(problem.pastlen(), problem.futurelen(), problem.colorspace())

# define the loss criterion
loss_fn = TVLoss(cspace)

hyperparams = {
    "lr": 0.01,
    "epochs": 3,
    "bsize": 64
}

# solve the problem with the model
solution, losses = problem.solve(model, loss_fn, hyperparams)

# solution statistics
movie(solution, devdirs[0])
diffplot(solution, devdirs[0])
