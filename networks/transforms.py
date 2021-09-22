
import numpy as np
import torch
import cv2


def dt(a):
    return cv2.distanceTransform((a * 255).astype(np.uint8), cv2.DIST_L2, 0)

def trimap_transform(trimap, L = 320):
    clicks = []
    for k in range(2):
        dt_mask = -dt(1 - trimap[:, :, k])**2
        clicks.append(np.exp(dt_mask / (2 * ((0.02 * L)**2))))
        clicks.append(np.exp(dt_mask / (2 * ((0.08 * L)**2))))
        clicks.append(np.exp(dt_mask / (2 * ((0.16 * L)**2))))
    clicks = np.array(clicks)
    return clicks

# For RGB !
imagenet_norm_std =  torch.from_numpy(np.array([0.229, 0.224, 0.225])).float().cuda()[None, :, None, None]
imagenet_norm_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).float().cuda()[None, :, None, None]


def normalise_image(image, mean=imagenet_norm_mean, std=imagenet_norm_std):
    return (image - mean) / std
