#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import numpy as np
from tqdm import tqdm
import sys
import torch
sys.path.append('/nmnt/media/home/alex_samoylenko/Federated/FederatedUNet/FederatedUNet')

from dataset.resources import get_datasets
from model.model import UNet
from updateWeights.update import LocalUpdate, average_weights
if __name__ == '__main__':
    epochs = 10

    start_time = time.time()

    torch.cuda.set_device(0)
    device = 'cuda'

    # load datasets
    datasets = get_datasets()

    # BUILD MODEL
    global_model = UNet(n_channels=1, n_classes=1).float()

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        for idx, dataset in enumerate(datasets):
            local_model = LocalUpdate(dataset=dataset)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
    #
    #     loss_avg = sum(local_losses) / len(local_losses)
    #     train_loss.append(loss_avg)
    #
    #     # Calculate avg training accuracy over all users at every epoch
    #     # list_acc, list_loss = [], []
    #     # global_model.eval()
    #     # for dataset in datasets:
    #     #     local_model = LocalUpdate(args=args, dataset=train_dataset,
    #     #                               idxs=user_groups[idx], logger=logger)
    #     #     acc, loss = local_model.inference(model=global_model)
    #     #     list_acc.append(acc)
    #     #     list_loss.append(loss)
    #     # train_accuracy.append(sum(list_acc)/len(list_acc))
    #
    #     # print global training loss after every 'i' rounds
    #     if (epoch+1) % print_every == 0:
    #         print(f' \nAvg Training Stats after {epoch+1} global rounds:')
    #         print(f'Training Loss : {np.mean(np.array(train_loss))}')
    #         print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
    #
    # # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)
    #
    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    #
    # # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)
    #
    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)
    #
    # print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
