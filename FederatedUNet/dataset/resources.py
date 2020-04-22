from .dataset import FederatedDataset, FederatedDataset_dpipe
import os
from dpipe.dataset.wrappers import cache_methods

def get_datasets():
    datasets = []
    meta_folder = '/nmnt/media/home/alex_samoylenko/Federated/FederatedUNet/FederatedUNet/dataset/metas'
    for meta_path in os.listdir(meta_folder):
        dataset = cache_methods(FederatedDataset_dpipe(os.path.join(meta_folder, meta_path)), methods=['load_x', 'load_y', 'len'])
        datasets.append(dataset)
    return datasets

