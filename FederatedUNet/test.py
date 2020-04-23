from dpipe.dataset import CSV
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset
from dpipe.dataset.wrappers import cache_methods
from dpipe.batch_iter import Infinite, load_by_random_id
from dpipe.batch_iter.utils import unpack_args


class FederatedDataset_dpipe(CSV):
    def __init__(self, meta_path, image_path='/nmnt/x3-hdd/data/DA/CC359/originalScaled',
                 target_path='/nmnt/x3-hdd/data/DA/CC359/Silver-standard-MLScaled',
                 img_col='img', target_col='target', index_col='id'):
        '''
        :param class_name: one of ['siemens-M', 'siemens-F', 'ge-M', 'ge-F', 'philips-M', 'philips-F']
        '''
        self.image_path = image_path
        self.target_path = target_path
        self.meta_path = meta_path
        self.img_col = img_col
        self.target_col = target_col
        self.df_meta = pd.read_csv(meta_path)
        self.ids = self.df_meta[index_col]

    def load_x(self, _id):
        img = np.load(os.path.join(self.image_path, self.df_meta.iloc[_id][self.img_col]))
        img = np.clip(img, -1000, None)
        return img.astype(np.float32)

    def load_y(self, _id):
        target = np.load(os.path.join(self.target_path, self.df_meta.iloc[_id][self.target_col]))
        return target.astype(int)


meta_path = '/nmnt/media/home/alex_samoylenko/Federated/FederatedUNet/FederatedUNet/dataset/metas/meta_philips-F.csv'
dataset = cache_methods(FederatedDataset_dpipe(meta_path),
                        methods=['load_x', 'load_y'])

n_samples_per_epoch = 10
batch_size = 2

trainloader = Infinite(
    load_by_random_id(dataset.load_x, dataset.load_y, ids=dataset.ids),
    batches_per_epoch=max(n_samples_per_epoch // batch_size, 1), batch_size=batch_size)

for i in trainloader():
    print(type(i))
    break