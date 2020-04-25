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


def inf_parse_args():
    parser = argparse.ArgumentParser(description='FederatedUNet inference parameters')

    parser.add_argument('--exp_path', default='/nmnt/media/home/alex_samoylenko/experiments/Federated')
    parser.add_argument('--exp_name')
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


def write_log(f, datasets, train_idxs, valid_idxs, test_idxs):
    for class_num in range(len(datasets)):
        f.write(datasets[class_num].class_name())  # class name
        f.write(str(train_idxs[class_num]))
        f.write(str(valid_idxs[class_num]))
        f.write(str(test_idxs[class_num]) + '\n')




