from torch.utils.data import Dataset
from torch import nn
from dpipe.batch_iter import Infinite, load_by_random_id
import copy
import torch
import numpy as np
from dpipe.batch_iter.utils import unpack_args
import time
from FederatedUNet.updateWeights.resources import *
from FederatedUNet.dataset.resources import get_random_slice

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
    def __init__(self, dataset, writer, args):
        self.args = args
        self.device = 'cuda'
        self.writer = writer
        self.class_name = dataset.class_name()
        self.dataset = dataset

        # criterion
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

    def train(self, model, train_idxs, local_lr, global_round):
        print(f'LR: {local_lr}, ROUND: {global_round}')
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=local_lr, weight_decay=1e-4)
        trainloader = Infinite(
            load_by_random_id(self.dataset.load_x, self.dataset.load_y, ids=train_idxs),
            unpack_args(get_random_slice),
            batches_per_epoch=max(self.args.n_samples_per_epoch // self.args.local_bs, 1),
            batch_size=self.args.local_bs)
        for _ in range(self.args.local_epochs):
            batch_loss = []
            for batch_idx, (image_slices, target_slices) in enumerate(trainloader()):
                print('batch')
                image_slices, target_slices = torch.tensor(image_slices), torch.tensor(target_slices)
                image_slices, target_slices = image_slices, target_slices.float()
                image_slices, target_slices = image_slices.to(self.device), target_slices.to(self.device)
                model.zero_grad()
                log_probs = model(image_slices)  # predictions
                torch.cuda.empty_cache()

                loss = self.criterion(log_probs, target_slices)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                torch.cuda.empty_cache()

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        trainloader.close()
        if global_round % self.args.show_every == 0:
            self.writer.add_figure(f'Training data: Input vs Prediction vs Target, {self.class_name}',  # visualize after last epoch
                                   visualize_preds(image_slices[0][0].cpu().detach().numpy(),
                                                   log_probs[0][0].cpu().detach().numpy(),
                                                   target_slices[0][0].cpu().detach().numpy()), global_step=global_round)
        return sum(epoch_loss) / len(epoch_loss) # model.state_dict()

    def inference(self, model, val_idxs):
        """ Returns the inference loss. """
        model.eval()

        inf_losses = []
        for val_idx in val_idxs:
            loss = 0
            image, target = self.dataset.load_x(val_idx), self.dataset.load_y(val_idx)
            image, target = np.moveaxis(np.array([image]), -1, 0), np.moveaxis(np.array([target]), -1, 0)
            image, target = torch.tensor(image), torch.tensor(target).float()
            for batch_no in range(image.shape[0] // self.args.local_bs):
                batch_image = image[batch_no * self.args.local_bs: (batch_no + 1) * self.args.local_bs].to(self.device)
                batch_target = target[batch_no * self.args.local_bs: (batch_no + 1) * self.args.local_bs].to(self.device)

                with torch.no_grad():
                    outputs = model(batch_image)
                    loss += self.criterion(outputs, batch_target).item()
                del batch_image, batch_target, outputs
                torch.cuda.empty_cache()

            # Inference
            inf_losses.append(loss / image.shape[0])

        return sum(inf_losses) / len(inf_losses)
