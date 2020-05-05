import torch
import os
import ast
import numpy as np

# import sys
# sys.path.insert(1, '/nmnt/media/home/alex_samoylenko/Federated/FederatedUNet')
from utils import inf_parse_args
from model.model import UNet
from dataset.resources import get_dataset, get_datasets


def main():
    args = inf_parse_args()

    try:
        os.mkdir(os.path.join(args.pred_save_path, args.exp_name))
        os.mkdir(os.path.join(args.pred_save_path, args.exp_name, 'valid'))
        os.mkdir(os.path.join(args.pred_save_path, args.exp_name, 'test'))
    except FileExistsError:
        print("Pred save path already exists")
        return
    # load model
    torch.cuda.set_device(0)
    device = 'cuda'
    model = UNet(n_channels=1, n_classes=1).float()
    weights = torch.load(os.path.join(args.exp_path, args.exp_name, 'model.pth'))
    model.load_state_dict(weights)
    model.eval()
    model.to(device)

    # load val and test ids
    train_idxs = []
    val_idxs = []
    test_idxs = []

    with open(os.path.join('/nmnt/media/home/alex_samoylenko/experiments/Federated', args.exp_name, 'IDX')) as f:
        while True:
            try:
                f.readline()  # name
                train_idxs.append(ast.literal_eval(f.readline().strip()))  # train_idx
                val_idxs.append(ast.literal_eval(f.readline().strip()))  # val_idx
                test_idxs.append(ast.literal_eval(f.readline().strip()))  # test_idx
                f.readline()  # newline
            except:
                break
    if not args.federated:
        datasets = get_dataset()
    else:
        datasets = get_datasets()

    # predict
    for dataset_num, dataset in enumerate(datasets):
        for val_idx in val_idxs[dataset_num]:
            outputs_stacked = []
            image = dataset.load_x(val_idx)
            image = np.moveaxis(np.array([image]), -1, 0)
            image = torch.tensor(image).float()
            for batch_no in range(image.shape[0] // args.bs + 1):
                batch_image = image[batch_no * args.bs: (batch_no + 1) * args.bs].to(device)
                if batch_image.shape[0] == 0:
                    break
                with torch.no_grad():
                    outputs = model(batch_image)
                outputs = outputs.cpu().numpy()
                outputs_stacked.extend(outputs)
                torch.cuda.empty_cache()

            outputs_stacked = np.stack(np.array(outputs_stacked), axis=-1)[0]
            np.save(os.path.join(args.pred_save_path, args.exp_name, 'valid', dataset.get_file_name(val_idx)), outputs_stacked)
        for test_idx in test_idxs[dataset_num]:
            outputs_stacked = []
            image = dataset.load_x(test_idx)
            image = np.moveaxis(np.array([image]), -1, 0)
            image = torch.tensor(image).float()
            for batch_no in range(image.shape[0] // args.bs + 1):
                batch_image = image[batch_no * args.bs: (batch_no + 1) * args.bs].to(device)
                if batch_image.shape[0] == 0:
                    break
                with torch.no_grad():
                    outputs = model(batch_image)
                outputs = outputs.cpu().numpy()
                outputs_stacked.extend(outputs)
                torch.cuda.empty_cache()

            outputs_stacked = np.stack(np.array(outputs_stacked), axis=-1)[0]
            np.save(os.path.join(args.pred_save_path, args.exp_name, 'test', dataset.get_file_name(test_idx)), outputs_stacked)


if __name__ == '__main__':
    main()