from dpipe.dataset import CSV
import pandas as pd
import os
import numpy as np
import cv2


class FederatedDataset(CSV):
    def __init__(self, meta_path, image_path = '/nmnt/x3-hdd/data/DA/CC359/originalScaled',
                 target_path='/nmnt/x3-hdd/data/DA/CC359/Silver-standard-MLScaled',
                 img_col='img', target_col='target', index_col='id'):
        '''
        :param class_name: one of ['siemens-M', 'siemens-F', 'ge-M', 'ge-F', 'philips-M', 'philips-F']
        '''
        self.image_path = image_path
        self.target_path = target_path
        self.meta_path = meta_path
        self._class_name = meta_path.split('_')[-1][:-4]
        self.img_col = img_col
        self.target_col = target_col
        self.df_meta = pd.read_csv(meta_path)
        self.ids = self.df_meta[index_col]

    def load_x(self, _id):
        img = np.load(os.path.join(self.image_path, self.df_meta.iloc[_id][self.img_col]))
        img = np.clip(img, -1000, 3000)
        img = cv2.resize(img, (256, 170))
        img = (img + 1000) / 4000
        return img.astype(np.float32)

    def load_y(self, _id):
        target = np.load(os.path.join(self.target_path, self.df_meta.iloc[_id][self.target_col]))
        target = cv2.resize(target, (256, 170))
        return target.astype(bool)

    def len(self):
        return self.df_meta.shape[0]

    def class_name(self):
        return self._class_name
