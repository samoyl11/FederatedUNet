from model.model import UNet
import torch
import socket
import pickle

model = UNet(n_channels=1, n_classes=1).float()
WEIGHTS_PATH = f'/nmnt/media/home/alex_samoylenko/experiments/Federated/no_federated/model.pth'
weights = torch.load(WEIGHTS_PATH)
model.load_state_dict(weights)

weight_pickle = pickle.dumps(weights)
print('weights ready')
sock = socket.socket()
sock.bind(('', 9091))
sock.listen(1)
conn, addr = sock.accept()

print ('connected:', addr)

while True:
    conn.send(weight_pickle)
    break
conn.close()
