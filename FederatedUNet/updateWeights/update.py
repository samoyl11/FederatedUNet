from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
from dpipe.batch_iter import Infinite, load_by_random_id
from dpipe.batch_iter.utils import unpack_args
import numpy as np
import copy
import sys
sys.path.append('/nmnt/media/home/alex_samoylenko/Federated/FederatedUNet')
from FederatedUNet.updateWeights.resources import *

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
    def __init__(self, dataset, writer):
        self.n_samples_per_epoch = 4
        self.local_lr = 1e-3
        self.local_epochs = 10
        self.local_bs = 2
        self.device = 'cuda'
        self.trainloader, self.validloader = self.train_val(dataset)
        self.writer = writer
        self.class_name = dataset.class_name()
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
            load_by_random_id(dataset.load_x, dataset.load_y, ids=idxs_val),
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
                # for img, target in zip(images, targets):
                #     self.writer.add_image(f'img_{self.class_name}', img)
                #     self.writer.add_image(f'target_{self.class_name}', target)
                images, targets = torch.tensor(images), torch.tensor(targets)
                images, targets = images, targets.float()
                images, targets = images.to(self.device), targets.to(self.device)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, targets)

                loss.backward()
                optimizer.step()
                self.writer.add_figure(f'predictions vs. actuals, {self.class_name}', visualize_preds(images[0][0].cpu().detach().numpy(),
                                                            log_probs[0][0].cpu().detach().numpy(), targets[0][0].cpu().detach().numpy()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference loss.
        """
        model.eval()
        inf_losses = []

        for batch_idx, (images, labels) in enumerate(self.validloader):
            images, targets = torch.tensor(images), torch.tensor(targets)
            images, targets = images, targets.float()
            images, targets = images.to(self.device), targets.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            inf_losses.append(batch_loss.item())

        return sum(inf_losses) / len(inf_losses)
