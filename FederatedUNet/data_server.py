import socket
import ast
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

from FederatedUNet.utils import data_server_parse_args, LrPolicy
from FederatedUNet.dataset.resources import get_dataset, get_train_val_test_idx
from FederatedUNet.model.model import UNet
from FederatedUNet.updateWeights.update import LocalUpdate

def main():
    args = data_server_parse_args()
    sock_meta = socket.socket()
    sock_meta.connect((args.aggregation_server, args.meta_port))
    data = sock_meta.recv(2048).decode()
    sock_meta.close()
    exp_path, exp_name, global_rounds, weighted = data.split('\n')
    args.exp_path = exp_path
    args.exp_name = exp_name
    args.global_rounds = int(global_rounds)
    args.weighted = int(weighted)

    writer = SummaryWriter(os.path.join(args.exp_path, args.exp_name, args.domain))

    dataset = get_dataset(args.domain)

    policy = ast.literal_eval(args.policy)
    lr_policy = LrPolicy(init_lr=args.init_lr, policy=policy)
    train_idx, valid_idx, test_idx = get_train_val_test_idx(dataset)

    # Save train/valid/test idxs
    with open(os.path.join(args.exp_path, args.exp_name, f'{args.domain}_idx'), 'w') as f:
        f.write(dataset.class_name() + '\n')  # class name
        f.write(str(train_idx) + '\n')
        f.write(str(valid_idx) + '\n')
        f.write(str(test_idx) + '\n\n')
        f.write(str(vars(args)))

    model = UNet(n_channels=1, n_classes=1).float()
    torch.cuda.set_device(args.device)
    device = 'cuda'
    model.to(device)

    info_str = str(len(args.domain)) + args.domain

    for global_round in tqdm(range(args.global_rounds)):
        #### receive model ####
        sock_recv = socket.socket()
        sock_recv.connect((args.aggregation_server, args.recv_port))
        data = b""
        while True:
            packet = sock_recv.recv(65536 * 2)
            if not packet:
                break
            data += packet

        sock_recv.close()
        weights = pickle.loads(data)
        model.load_state_dict(weights)

        local_model = LocalUpdate(dataset=dataset, writer=writer, args=args)

        #### valid loss ####
        if global_round > 0:
            valid_loss = local_model.inference(model=model, val_idxs=valid_idx)
            writer.add_scalar('Valid loss', valid_loss, global_step=global_round - 1)
            if args.weighted:
                sock_send = socket.socket()
                sock_send.connect((args.aggregation_server, args.send_port))
                sock_send.send(info_str.encode())
                sock_send.send(str(valid_loss).encode())
                sock_send.close()

        #### update model ####
        train_loss = local_model.train(model=model,
                                    train_idxs=train_idx,
                                    local_lr=lr_policy.lr,
                                    global_round=global_round)
        writer.add_scalar('Train loss', train_loss, global_step=global_round)

        # Lr step
        lr_policy.step()

        #### send updated model to aggregation server ####
        print('finished local training...')
        sock_send = socket.socket()
        sock_send.connect((args.aggregation_server, args.send_port))
        sock_send.send(info_str.encode())
        sock_send.send(pickle.dumps(model.state_dict()))
        sock_send.close()

    writer.close()

if __name__ == '__main__':
    main()
