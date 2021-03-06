import pathlib
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from tqdm import tqdm
from visionutils import flow2mag

# workaround for bug https://github.com/tqdm/tqdm/issues/481
tqdm.monitor_interval = 0

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 50}

plt.rc('font', **font)

def movie(solution, rundir):
    # directory where to save the movie
    paths = rundir.split('/')
    paths[0] = "movies"
    moviedir = '/'.join(paths)

    # directory where to save the optical flow
    paths[0] = "flows"
    flowdir = '/'.join(paths)

    # create directory if it does not exist
    pathlib.Path(moviedir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(flowdir).mkdir(parents=True, exist_ok=True)

    # setup progress bar
    nimgs = len(listdir(rundir))
    progress = tqdm(total=nimgs)

    # save original and predicted frames
    for t, (imgtrue, imghat) in enumerate(solution.play(rundir)):
        # convert from torch to matplotlib format
        if imgtrue.shape[0] == 3: # RGB
            imgtrue = imgtrue.transpose([1,2,0])
            imghat  = imghat.transpose([1,2,0])
        if imgtrue.shape[0] == 2: # FLOW
            flowtrue = np.copy(imgtrue.transpose([1,2,0]))
            flowhat  = np.copy(imghat.transpose([1,2,0]))
            np.save(flowdir+"/{:04}.npy".format(t+1), flowhat)
            imgtrue = flow2mag(flowtrue)
            imghat  = flow2mag(flowhat)
        else:
            imgtrue = imgtrue[0,:,:]
            imghat  = imghat[0,:,:]

        fig, ax = plt.subplots(1,2, figsize=(20,20))
        plt.subplot(1,2,1)
        plt.imshow(imgtrue, cmap="binary_r")
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.axis("off")
        plt.title("original", fontsize=50)
        plt.subplot(1,2,2)
        plt.imshow(imghat, cmap="binary_r")
        plt.axis("off")
        plt.title("neural network", fontsize=50)
        plt.annotate("time {:04}".format(t+1), xy=(.01,.92), xycoords="figure fraction")
        plt.tight_layout()
        plt.savefig(moviedir+"/{:04}.png".format(t+1), bbox_inches="tight")
        plt.close()

        progress.update()

def diffplot(solution, rundir):
    # directory name for saving the diff plot
    paths = rundir.split('/')
    paths[0] = "diffplots"
    diffdir = '/'.join(paths)

    # create directory if it does not exist
    pathlib.Path(diffdir).mkdir(parents=True, exist_ok=True)

    trues, fakes = [], []
    for (imgtrue, imghat) in solution.play(rundir):
        trues.append(imgtrue)
        fakes.append(imghat)

    dtrues = np.diff(trues)
    dfakes = np.diff(fakes)

    dtrues = [np.sum(np.abs(d)) for d in dtrues]
    dfakes = [np.sum(np.abs(d)) for d in dfakes]

    X = np.array([dtrues, dfakes]).T
    np.savetxt(diffdir+"/plot.dat", X, header="1st column = original, 2nd column = neural network")

    fig = plt.figure(figsize=(20,20))
    plt.plot(dtrues/dtrues[0], label="original")
    plt.plot(dfakes/dfakes[0], label="neural network")
    plt.xlabel("time step")
    plt.ylabel("normalized difference")
    plt.legend()
    plt.savefig(diffdir+"/plot.png", bbox_inches="tight")
    plt.close()
