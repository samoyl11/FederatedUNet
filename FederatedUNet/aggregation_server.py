import copy
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import ast
import sys
sys.path.append('/nmnt/media/home/alex_samoylenko/Federated/FederatedUNet')

from FederatedUNet.dataset.resources import get_datasets, get_random_idxs, get_dataset
from FederatedUNet.model.model import UNet
from FederatedUNet.updateWeights.update import LocalUpdate, average_weights
from FederatedUNet.utils import train_parse_args, LrPolicy


def main():
    args = train_parse_args()
    writer = SummaryWriter(os.path.join(args.exp_path, args.exp_name))
    torch.cuda.set_device(0)
    device = 'cuda'

    # learning rate policy
    policy = ast.literal_eval(args.policy)
    lr_policy = LrPolicy(init_lr=args.lr, policy=policy)

    # load datasets
    if args.federated:
        datasets = get_datasets()
    else:
        datasets = get_dataset()
    train_idxs, valid_idxs, test_idxs = get_random_idxs(datasets)

    # BUILD MODEL
    global_model = UNet(n_channels=1, n_classes=1).float()

    # Set the model to train and send it to device.
    global_model.to(device)

    # weights
    global_weights = global_model.state_dict()

    # Training
    for global_round in tqdm(range(args.global_rounds)):
        local_weights, local_train_losses_global_round, local_val_losses_global_round = [], [], []
        print(f'\n | Global Training Round : {global_round+1} |\n')

        global_model.train()
        for ds_num, dataset in enumerate(datasets):
            local_model = LocalUpdate(dataset=dataset, writer=writer, args=args)
            w, loss = local_model.train(model=global_model,
                                        train_idxs=train_idxs[ds_num],
                                        local_lr=lr_policy.lr,
                                        global_round=global_round)
            local_weights.append(copy.deepcopy(w))
            local_train_losses_global_round.append(copy.deepcopy(loss))
        writer.add_scalars('Global round train losses',
                           {datasets[i].class_name(): local_train_losses_global_round[i] for i in range(len(local_train_losses_global_round))},
                           global_step=global_round)
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # Lr step
        writer.add_scalar('Learning rate', lr_policy.lr, global_step=global_round)
        lr_policy.step()

        # Calculate valid loss
        global_model.eval()
        if global_round % args.show_every == 0:
            for ds_num, dataset in enumerate(datasets):
                local_model = LocalUpdate(dataset=dataset, writer=writer, args=args)
                loss = local_model.inference(model=global_model, val_idxs=valid_idxs[ds_num])
                local_val_losses_global_round.append(loss)
            writer.add_scalars('Global round val losses',
                               {datasets[i].class_name(): local_val_losses_global_round[i] for i in range(len(local_val_losses_global_round))},
                               global_step=global_round)

    # Save train/valid/test idxs
    with open(os.path.join(args.exp_path, args.exp_name, 'IDX'), 'w') as f:
        for class_num in range(len(datasets)):
            f.write(datasets[class_num].class_name() + '\n')  # class name
            f.write(str(train_idxs[class_num]) + '\n')
            f.write(str(valid_idxs[class_num]) + '\n')
            f.write(str(test_idxs[class_num]) + '\n\n')
        f.write(str(vars(args)))
    # Save model
    torch.save(global_model.state_dict(), os.path.join(args.exp_path, args.exp_name, 'model.pth'))

    writer.close()


if __name__ == '__main__':
    main()