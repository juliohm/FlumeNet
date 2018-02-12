import os, errno
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 50}

plt.rc('font', **font)

def movie(solution, rundir):
    # directory name for saving the movie
    paths = rundir.split('/')
    paths[0] = "movies"
    moviedir = '/'.join(paths)

    # create directory if it does not exist
    try:
        os.makedirs(moviedir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # save original and predicted frames
    for t, (imgtrue, imghat) in enumerate(solution.play(rundir)):
        # convert from torch to matplotlib format
        if imgtrue.shape[0] == 3: # RGB
            imgtrue = imgtrue.transpose([1,2,0])
            imghat  = imghat.transpose([1,2,0])
        else:
            imgtrue = imgtrue[0,:,:]
            imghat  = imghat[0,:,:] > 0.5

        fig, ax = plt.subplots(1,2, figsize=(20,20))
        plt.subplot(1,2,1)
        plt.imshow(imgtrue)
        plt.gca().axes.xaxis.set_ticklabels([])
        plt.gca().axes.yaxis.set_ticklabels([])
        plt.axis("off")
        plt.title("original", fontsize=50)
        plt.subplot(1,2,2)
        plt.imshow(imghat)
        plt.axis("off")
        plt.title("neural network", fontsize=50)
        plt.annotate("time {:04}".format(t+1), xy=(.01,.92), xycoords="figure fraction")
        plt.tight_layout()
        plt.savefig(moviedir+"/{:04}.png".format(t+1), bbox_inches="tight")
        plt.close()
