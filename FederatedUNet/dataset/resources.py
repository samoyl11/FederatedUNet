from FederatedUNet.dataset.dataset import FederatedDataset
import os
from dpipe.dataset.wrappers import cache_methods
from numpy.random import shuffle
import numpy as np

meta_folder = '/nmnt/media/home/alex_samoylenko/Federated/FederatedUNet/FederatedUNet/dataset/metas'

def get_datasets():
    datasets = []
    for meta_path in os.listdir(meta_folder):
        dataset = cache_methods(FederatedDataset(os.path.join(meta_folder, meta_path)), methods=['load_x', 'load_y', 'len',
                                                                                                 'class_name', 'get_file_name'])
        datasets.append(dataset)
    return datasets


def get_dataset(domain_name):
    meta_name = os.path.join(meta_folder, f'meta_{domain_name}.csv')
    dataset = cache_methods(FederatedDataset(meta_name), methods=['load_x', 'load_y', 'len', 'class_name', 'get_file_name'])
    return dataset

def get_full_dataset():
    full_meta_path = '/nmnt/media/home/alex_samoylenko/Federated/FederatedUNet/FederatedUNet/dataset/meta.csv'
    dataset = cache_methods(FederatedDataset(full_meta_path), methods=['load_x', 'load_y', 'len', 'class_name', 'get_file_name'])
    return dataset


def get_train_val_test_idx(dataset):
    idx = list(range(dataset.len()))
    shuffle(idx)
    train_idx = idx[:int(0.8 * dataset.len())]
    valid_idx = idx[int(0.8 * dataset.len()):int(0.9 * dataset.len())]
    test_idx = idx[int(0.9 * dataset.len()):]
    return train_idx, valid_idx, test_idx


def get_random_slice(*inputs):
    img, target = inputs
    _id = np.random.choice(img.shape[-1])

    return [[img[..., _id]], [target[..., _id]]]