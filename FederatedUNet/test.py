import argparse

def train_parse_args():
    parser = argparse.ArgumentParser(description='FederatedUNet training parameters')

    parser.add_argument('--global_rounds', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--local_epochs', type=int, default=8)
    parser.add_argument('--local_bs', type=int, default=6)
    parser.add_argument('--policy', default="{5: 0.5, 10: 0.5, 15: 0.5, 20: 0.5, 23: 0.25, 26: 0.25}")
    parser.add_argument('--exp_path', default='/nmnt/media/home/alex_samoylenko/experiments/Federated')
    parser.add_argument('--exp_name')
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_samples_per_epoch', type=int, default=5000)
    parser.add_argument('--show_every', type=int, default=5)

    args = parser.parse_args()
    return args

parsed = vars(train_parse_args())
print(str(parsed))