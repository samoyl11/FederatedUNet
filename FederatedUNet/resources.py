from dpipe.dataset import CSV
import pandas as pd
import os
import numpy as np

class FederatedDataset(CSV):
    def __init__(self, image_path, target_path, meta_path, img_col='img', target_col='target'):
        '''
        :param class_name: one of ['siemens-M', 'siemens-F', 'ge-M', 'ge-F', 'philips-M', 'philips-F']
        '''
        self.image_path = image_path
        self.target_path = target_path
        self.meta_path = meta_path
        self.img_col = img_col
        self.target_col = target_col
        self.df_meta = pd.read_csv(meta_path)

    def load_x(self, _id):
        img = np.load(os.path.join(self.image_path, self.df_meta.iloc[_id][self.img_col]))
        img = np.clip(-1000)
        return img.astype(np.float32)
    def load_y(self, _id):
        target = np.load(os.path.join(self.target_path, self.df_meta.iloc[_id][self.target_col]))
        return target.astype(int)
