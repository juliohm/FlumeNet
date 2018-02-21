from problems import VideoGenProblem
from netmodels import CodeRegNet, TorricelliNet
from torch.nn import MSELoss, BCELoss, L1Loss
from plotting import movie, diffplot

# color space
cspace = "BW"
prefix = "data/bw/" if cspace == "BW" else "data/rgb/"

# datasets
traindirs = ["Run 1", "Run 2 - 1", "Run 2 - 2", "Run 3 - 2", "Run 4", "Run 5",
             "Run 6 - 1", "Run 6 - 2", "Run 7 - 1", "Run 7 - 2"]
devdirs   = ["Run 3 - 1"]

traindirs = [prefix+tdir for tdir in traindirs]
devdirs   = [prefix+ddir for ddir in devdirs]

# define the problem
problem = VideoGenProblem(traindirs, devdirs, cspace=cspace, pinds=[1,2,3], finds=[4])

# define the network model
# model = CodeRegNet(problem.pastlen(), problem.futurelen(), problem.colorspace())
model = TorricelliNet(problem.pastlen(), problem.futurelen(), problem.colorspace())

# define the loss criterion
loss_fn = MSELoss() if problem.colorspace() == "RGB" else BCELoss()

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
