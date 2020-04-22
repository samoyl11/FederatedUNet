from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class FederatedDataset(Dataset):
    def __init__(self, data_path, target_path, data_class):
        '''
        :param data_class: one of ['siemens-M', 'siemens-F', 'ge-M', 'ge-F', 'philips-M', 'philips-F']
        '''
        self.data_path = data_path
        self.target_path = target_path
        self.data_class = data_class
        brand, filterUsed = data_class.split('-')

        # TODO CHECK NAMES MATCHING
        self.img_names = sorted(list(filter(lambda x: (brand in x) & (filterUsed in x), os.listdir(data_path))))[:15]
        self.target_names = sorted(list(filter(lambda x: (brand in x) & (filterUsed in x), os.listdir(target_path))))[:15]
        assert (len(self.img_names) == len(self.target_names))

        self.imgs = [np.load(os.path.join(data_path, img_name)) for img_name in self.img_names]
        self.targets = [np.load(os.path.join(target_path, target_name)) for target_name in self.target_names]

    def __getitem__(self, index):
        print(f'getting img named {self.img_names[index]} and target named {self.target_names[index]}')
        return self.imgs[index], self.targets[index]

    def __len__(self):
        return len(self.img_names)


dataset = FederatedDataset('/nmnt/x3-hdd/data/DA/CC359/originalScaled', '/nmnt/x3-hdd/data/DA/CC359/Silver-standard-MLScaled', 'siemens-F')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
dataiter = iter(dataloader)
images, labels = dataiter.next()
images, labels = dataiter.next()
images, labels = dataiter.next()
images, labels = dataiter.next()
images, labels = dataiter.next()

print(images[0].shape, labels[0].shape, images[1].shape, labels[1].shape, type(images[0]))
