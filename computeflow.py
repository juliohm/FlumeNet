import os
import numpy as np
import pathlib
from imageio import imread
import cv2

indir = 'data/gray/'
outdir = 'data/flow/'
hsvdir = 'data/flowhsv/'
warpdir = 'data/warped/'

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

for rundir in sorted(os.listdir(indir)):
    infulldir = indir+rundir
    outfulldir = outdir+rundir
    hsvfulldir = hsvdir+rundir
    warpfulldir = warpdir+rundir
    infnames = [infulldir+"/"+fname for fname in sorted(os.listdir(infulldir))]
    outfnames = [outfulldir+"/"+fname[:-4]+".npy" for fname in sorted(os.listdir(infulldir))]
    hsvfnames = [hsvfulldir+"/"+fname for fname in sorted(os.listdir(infulldir))]
    warpfnames = [warpfulldir+"/"+fname for fname in sorted(os.listdir(infulldir))]

    # create output directory
    pathlib.Path(outfulldir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(hsvfulldir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(warpfulldir).mkdir(parents=True, exist_ok=True)

    for t in range(len(infnames)-1):
        im1 = np.asarray(imread(infnames[t]))
        im2 = np.asarray(imread(infnames[t+1]))

        flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # save flow to numpy file
        np.save(outfnames[t], flow)

        # draw flow in HSV (magnitude and angle)
        flowhsv = draw_hsv(flow)
        cv2.imwrite(hsvfnames[t], flowhsv)

        im2w = warp_flow(im1, flow)
#         im2w = 255.*(im2w > 128) # make sure the image is binary
        cv2.imwrite(warpfnames[t], im2w)

# print min/max flow for normalization during training
IMIN =  np.inf
IMAX = -np.inf
for rundir in sorted(os.listdir(outdir)):
    d = outdir+rundir
    for fname in sorted(os.listdir(d)):
        ffname = d+"/"+fname
        img = np.load(ffname)
        imin, imax = img.min(), img.max()
        if imin < IMIN:
            IMIN = imin
        if imax > IMAX:
            IMAX = imax

print(IMIN, IMAX)
