import argparse
import pickle

def aggregation_server_parse_args():
    parser = argparse.ArgumentParser(description='FederatedUNet aggregation server parameters')

    parser.add_argument('--exp_name')
    parser.add_argument('--exp_path', default='/nmnt/media/home/alex_samoylenko/experiments/Federated')
    parser.add_argument('--global_rounds', type=int, default=50)
    parser.add_argument('--send_port', type=int, default=9150)
    parser.add_argument('--recv_port', type=int, default=9152)
    parser.add_argument('--meta_port', type=int, default=9154)
    parser.add_argument('--weighted', type=int, default=0)
    parser.add_argument('--domains', type=int, default=6)
    # parser.add_argument('--federated', type=int, default=1)
    args = parser.parse_args()
    return args


def data_server_parse_args():
    parser = argparse.ArgumentParser(description='FederatedUNet data server parameters')

    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--local_epochs', type=int, default=8)
    parser.add_argument('--local_bs', type=int, default=10)
    parser.add_argument('--policy', default="{25:0.5,37:0.5,42:0.5,46:0.1}")  # {5: 0.5, 10: 0.5, 15: 0.5, 20: 0.5, 23: 0.25, 26: 0.25}
    parser.add_argument('--n_samples_per_epoch', type=int, default=2000)
    parser.add_argument('--aggregation_server', default='neuro-t')
    parser.add_argument('--recv_port', type=int, default=9150)
    parser.add_argument('--send_port', type=int, default=9152)
    parser.add_argument('--meta_port', type=int, default=9154)
    parser.add_argument('--show_every', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--domain', type=str, required=True)  # ge-F
    # parser.add_argument('--federated', type=int, default=1)
    args = parser.parse_args()
    return args


def inf_parse_args():
    parser = argparse.ArgumentParser(description='FederatedUNet inference parameters')

    parser.add_argument('--exp_path', default='/nmnt/media/home/alex_samoylenko/experiments/Federated')
    parser.add_argument('--exp_name')
    parser.add_argument('--bs', type=int, default=20)
    parser.add_argument('--pred_save_path', default='/nmnt/x3-hdd/data/Federated')
    parser.add_argument('--federated', type=int, default=1)
    # parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    return args


class LrPolicy:
    def __init__(self, init_lr, policy):
        self.lr = init_lr
        self.policy = policy # dict of epoch : multiplier
        self.count = 1

    def step(self):
        self.count += 1
        if self.count in self.policy.keys():
            self.lr *= self.policy[self.count]



def send_weights(conn, weights, socket_lock):
    conn.send(weights)
    socket_lock.release()
    conn.close()


def recv_weights(conn, weights_dict, socket_lock):
    dom_name_len = conn.recv(1)
    dom_name_len = int(dom_name_len.decode())
    dom_name = conn.recv(dom_name_len)
    weight = conn.recv(2048)
    socket_lock.release()
    conn.close()
    print(f'Received weight from {dom_name}')
    weight = float(weight.decode())
    weights_dict[dom_name.decode()] = weight

def recv_model_weights(conn, model_weights_dict, socket_lock):
    dom_name_len = conn.recv(1)
    dom_name_len = int(dom_name_len.decode())
    dom_name = conn.recv(dom_name_len)
    data = b""
    while True:
        packet = conn.recv(65536 * 2)
        if not packet:
            break
        data += packet
    socket_lock.release()
    conn.close()
    print(f'Received model weights from {dom_name}')
    weights = pickle.loads(data)
    model_weights_dict[dom_name.decode()] = weights


def send_meta(conn, args, socket_lock):
    meta_string = f"{args.exp_path}\n{args.exp_name}\n{args.global_rounds}\n{args.weighted}"
    conn.send(meta_string.encode())
    socket_lock.release()
    conn.close()


