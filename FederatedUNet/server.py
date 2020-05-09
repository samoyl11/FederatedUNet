from model.model import UNet
import torch
import socket
import pickle

# import thread module
from _thread import *
import threading
import os
import time
import copy

print_lock = threading.Lock()


def send_model(c, weights):
    c.send(weights)
    c.close()

def recv_model(c, weights_list):
    data = b""
    while True:
        packet = c.recv(65536 * 2)
        if not packet:
            break
        data += packet
    c.close()
    weights = pickle.loads(data)
    weights_list.append(weights)

SAVE_PATH = '/nmnt/media/home/alex_samoylenko/checkpoints'

model = UNet(n_channels=1, n_classes=1).float()
WEIGHTS_PATH = f'/nmnt/media/home/alex_samoylenko/experiments/Federated/no_federated/model.pth'
weights = torch.load(WEIGHTS_PATH)
model.load_state_dict(weights)
global_counter_1 = 0
global_counter_2 = 0
domains_num = 2
# Save model

weight_pickle = pickle.dumps(weights)
print('weights ready')
sock = socket.socket()
sock.bind(('', 9091))
sock.listen(5)


#send
while global_counter_1 < domains_num:
    # lock acquired by client
    c, addr = sock.accept()
    global_counter_1 += 1
    # print_lock.acquire()
    print('Connected to :', addr[0], ':', addr[1])

    # Start a new thread and return its identifier
    start_new_thread(send_model, (c, copy.deepcopy(weight_pickle)))


updated_models = []
#recv
while global_counter_2 < domains_num:
    # lock acquired by client
    c, addr = sock.accept()
    global_counter_2 += 1
    # print_lock.acquire()
    print('Connected to :', addr[0], ':', addr[1])

    # Start a new thread and return its identifier
    start_new_thread(recv_model, (c, updated_models))
time.sleep(15)
print(len(updated_models))




