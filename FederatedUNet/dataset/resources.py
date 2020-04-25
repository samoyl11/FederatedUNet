from .dataset import FederatedDataset
import os
from dpipe.dataset.wrappers import cache_methods
from numpy.random import shuffle
import numpy as np


def get_datasets():
    datasets = []
    meta_folder = '/nmnt/media/home/alex_samoylenko/Federated/FederatedUNet/FederatedUNet/dataset/metas'
    for meta_path in os.listdir(meta_folder):
        dataset = cache_methods(FederatedDataset(os.path.join(meta_folder, meta_path)), methods=['load_x', 'load_y', 'len', 'class_name'])
        datasets.append(dataset)
    return datasets


def get_random_idxs(datasets):
    train_idx, valid_idx, test_idx = [], [], []
    for dataset in datasets:
        idxs = list(range(dataset.len()))
        shuffle(idxs)
        train_idx.append(idxs[:int(0.8 * dataset.len())])
        valid_idx.append(idxs[int(0.8 * dataset.len()):int(0.9 * dataset.len())])
        test_idx.append(idxs[int(0.8 * dataset.len()):int(0.9 * dataset.len())])
    return train_idx, valid_idx, test_idx


def get_random_slice(*inputs):
    img, target = inputs
    _id = np.random.choice(img.shape[-1])

    return [[img[..., _id]], [target[..., _id]]]