from matplotlib import pyplot as plt
import torch
import numpy as np
import copy

def img_target_show(img, target):
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(img, cmap='Greys')
    plt.subplot(122)
    plt.imshow(target)


def visualize_preds(img, pred, target):
    fig = plt.figure(figsize=(24, 8))
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(pred)
    plt.subplot(133)
    plt.imshow(target)
    return fig

def get_slice(*inputs):
    img, target = inputs
    _id = np.random.choice(img.shape[-1])

    return [[img[..., _id]], [target[..., _id]]]


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg