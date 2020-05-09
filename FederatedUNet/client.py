from model.model import UNet
import torch
import socket
import pickle
import time
model = UNet(n_channels=1, n_classes=1).float()

start = time.time()
sock = socket.socket()
sock.connect(('neuro-t', 9091))

data = b""

while True:
    packet = sock.recv(65536 * 2)
    if not packet:
        break
    data += packet
sock.close()
print('received all')
print(time.time() - start)
weights = pickle.loads(data)
model.load_state_dict(weights)

print('DOING SOMETHING')
time.sleep(30)

sock = socket.socket()
sock.connect(('neuro-t', 9091))
sock.send(pickle.dumps(weights))
sock.close()


