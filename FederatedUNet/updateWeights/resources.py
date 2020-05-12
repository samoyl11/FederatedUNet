from matplotlib import pyplot as plt
import torch
import copy

def visualize_preds(img, pred, target):
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(pred)
    plt.subplot(133)
    plt.imshow(target)
    return fig

def average_weights(model_weights_dict):
    """
    Returns the average of the weights.
    """
    keys = list(model_weights_dict.keys())
    w_avg = copy.deepcopy(model_weights_dict[keys[0]])

    for key in w_avg.keys():
        for i in range(1, len(keys)):
            w_avg[key] += model_weights_dict[keys[i]][key]
        w_avg[key] = torch.div(w_avg[key], len(keys))
    return w_avg

def weighted_average(model_weights_dict, weights_dict):

    keys = list(model_weights_dict.keys())
    w_avg = copy.deepcopy(model_weights_dict[keys[0]])

    for key in w_avg.keys():
        w_avg[key] = 0

    for key in w_avg.keys():
        for i in range(len(keys)):
            w_avg[key] += torch.mul(model_weights_dict[keys[i]][key], weights_dict[keys[i]])
    return w_avg