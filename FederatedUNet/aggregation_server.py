import copy
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import socket
import numpy as np
import pickle
from _thread import start_new_thread
from threading import Lock

from FederatedUNet.model.model import UNet
from FederatedUNet.updateWeights.update import LocalUpdate, average_weights, weighted_average
from FederatedUNet.utils import aggregation_server_parse_args, LrPolicy, send_weights, recv_model_weights, recv_weights, send_meta

def main():
    args = aggregation_server_parse_args()
    writer = SummaryWriter(os.path.join(args.exp_path, args.exp_name, 'aggregation'))
    socket_lock = Lock()
    #### build model ####
    global_model = UNet(n_channels=1, n_classes=1).float()
    global_weights = global_model.state_dict()

    sock_send = socket.socket()
    sock_send.bind(('', args.send_port))
    sock_send.listen(args.domains)

    sock_recv = socket.socket()
    sock_recv.bind(('', args.recv_port))
    sock_recv.listen(args.domains)

    sock_meta= socket.socket()
    sock_meta.bind(('', args.meta_port))
    sock_meta.listen(args.domains)

    meta_cnt = 0
    while meta_cnt < args.domains:
        conn, _ = sock_meta.accept()
        meta_cnt += 1
        # Start a new thread to send weights
        socket_lock.acquire()
        start_new_thread(send_meta, (conn, args, socket_lock))
    sock_meta.close()
    socket_lock.acquire()
    #### wait until sending is complete
    socket_lock.release()

    #### Training ####
    for global_round in tqdm(range(args.global_rounds)):
        print(f'\n | Global Training Round : {global_round + 1} |\n')
        send_cnt = 0
        weights_cnt = 0
        recv_cnt = 0
        local_model_weights_dict = dict()
        weights_dict = dict()

        #### model weights to binary ####
        pickled_weights = pickle.dumps(global_weights)

        #### send weights to data servers ####
        while send_cnt < args.domains:
            conn, _ = sock_send.accept()
            send_cnt += 1
            # Start a new thread to send weights
            socket_lock.acquire()
            start_new_thread(send_weights, (conn, copy.deepcopy(pickled_weights), socket_lock))

        socket_lock.acquire()
        #### wait until sending is complete
        socket_lock.release()

        #### receive averaging weights from data servers ####
        if args.weighted:
            if global_round > 0:
                while weights_cnt < args.domains:
                    conn, _ = sock_recv.accept()
                    weights_cnt += 1
                    # Start a new thread to send weights
                    socket_lock.acquire()
                    start_new_thread(recv_weights, (conn, weights_dict, socket_lock))

        #### receive model weights from data servers ####

        while recv_cnt < args.domains:
            conn, _ = sock_recv.accept()
            recv_cnt += 1
            # Start a new thread to send weights
            socket_lock.acquire()
            start_new_thread(recv_model_weights, (conn, local_model_weights_dict, socket_lock))

        #### update global weights ####
        socket_lock.acquire()
        if args.weighted:
            if global_round > 0:
                sum_losses = sum(weights_dict.values())
                for key in weights_dict:
                    weights_dict[key] /= sum_losses
                print('weights: ', weights_dict)
                global_weights = weighted_average(local_model_weights_dict, weights_dict)
            else:
                global_weights = average_weights(local_model_weights_dict)
        else:
            global_weights = average_weights(local_model_weights_dict)

        global_model.load_state_dict(global_weights)
        socket_lock.release()

    sock_recv.close()
    sock_send.close()
    # Save model
    torch.save(global_model.state_dict(), os.path.join(args.exp_path, args.exp_name, 'model.pth'))

    writer.close()


if __name__ == '__main__':
    main()