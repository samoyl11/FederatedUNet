from model.model import UNet
import torch
import socket
import pickle

model = UNet(n_channels=1, n_classes=1).float()
sock = socket.socket()
sock.connect(('neuro-x5', 9091))

data = b""

while True:
    packet = sock.recv(65536)
    if not packet:
        break
    data += packet
sock.close()

weights = pickle.loads(data)
model.load_state_dict(weights)

print(model)
