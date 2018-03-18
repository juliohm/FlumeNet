import numpy as np
import torch
import cv2

def flow2mag(flow):
    """
    Convert optical flow (velocity in x and y) to magnitude
    and angle and save it in HSV colorspace for visualization.
    """
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

def flowbatch2mag(flowbatch):
    """
    Process a batch of flow and convert it to magnitude and angle.
    """
    n = flowbatch.size(0)
    mags = []
    for i in range(n):
        flow = flowbatch[i,:,:,:].cpu().numpy().transpose([1, 2, 0])
        mag = flow2mag(flow).transpose([2, 0, 1])
        mag = mag[np.newaxis,...]
        mags.append(mag)

    mags = np.concatenate(mags, axis=0)
    return torch.Tensor(mags)
