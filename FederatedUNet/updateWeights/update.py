from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
from dpipe.batch_iter import Infinite, load_by_random_id
from dpipe.batch_iter.utils import unpack_args
import numpy as np
import copy

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class LocalUpdate:
    def __init__(self, dataset):
        self.n_samples_per_epoch = 4
        self.local_lr = 1e-3
        self.local_epochs = 10
        self.local_bs = 2
        self.device = 'cuda'
        self.trainloader, self.validloader = self.train_val(dataset)

        print('init')
        # optimizer and criterion
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

    def train_val(self, dataset):
        """
        Returns train, validation and test dataloaders for a given dataset
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs = list(range(dataset.len()))
        idxs_train = idxs[:int(0.9 * len(idxs))]
        idxs_val = idxs[int(0.9 * len(idxs)):]

        train_loader = Infinite(
                load_by_random_id(dataset.load_x, dataset.load_y, ids=idxs_train),
                unpack_args(get_slice),
                batches_per_epoch=max(self.n_samples_per_epoch//self.local_bs,1), batch_size=self.local_bs)
        val_loader = Infinite(
            load_by_random_id(dataset.load_x, dataset.load_y, ids=idxs_train),
            unpack_args(get_slice),
            batches_per_epoch=max(self.n_samples_per_epoch // self.local_bs, 1), batch_size=self.local_bs)
        return train_loader, val_loader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=self.local_lr, weight_decay=1e-4)

        for iter in range(self.local_epochs):
            batch_loss = []
            for batch_idx, (images, targets) in enumerate(self.trainloader()):
                print(images[0].dtype)
                images, targets = torch.tensor(images), torch.tensor(targets)
                images, targets = images.float(), targets.float()
                images, targets = images.to(self.device), targets.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, targets)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)



def get_slice(*inputs):
    img, target = inputs
    # masked = ((x < 0).sum(axis=-2).sum(axis=-2) != 0)
    # to_decide = np.nonzero(masked)[0]
    _id = np.random.choice(img.shape[-1])

    #     print("X: ", x[:, _id, ...].shape, _id)
    #     print("M: ", m[:, _id, ...].shape, _id)
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