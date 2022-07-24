
import argparse
from argparse import Namespace
import os
import torch
import numpy as np
import torch.nn as nn
import torch
from torchvision import utils
from datasets import *

# models
from models.LSA_mnist import LSA_MNIST
from models.LSA_cifar10 import LSA_CIFAR10
from models.transform_realnvp import TinvREALNVP
from models.glow_models import Glow
import matplotlib.pyplot as plt

# random seed
from utils import set_random_seed
from result_helpers.ood_trainer import OODTrainer
# path
from utils import create_checkpoints_dir

import time

def _init_fn():
    np.random.seed(12)

def setDataset(ds_name, n_class, ind, train_intra, seg_len):
    if ds_name == 'mnist':
        dataset = MNIST(path='data/MNIST', n_class=n_class, InD=ind)
    elif ds_name == 'fmnist':
        dataset = FMNIST(path='data/FMNIST', n_class=n_class, InD=ind, train_intra=train_intra)
    elif ds_name == 'kmnist':
        dataset = KMNIST(path='data/KMNIST', InD=ind)
    elif ds_name == 'omniglot':
        dataset = OMNIGLOT(path='data/OMNIGLOT', InD=ind)
    elif ds_name == 'ecg':
        dataset = ECG(path='data/PSG', InD=ind, seg_len=seg_len)
    elif ds_name == 'eeg':
        dataset = EEG(path='data/PSG', InD=ind, seg_len=seg_len)
    elif ds_name == 'cifar10':
        dataset = CIFAR10(path='data/CIFAR10', n_class=n_class, InD=ind, train_intra=train_intra)
    elif ds_name == 'svhn':
        dataset = SVHN(path='data/SVHN', n_class=n_class, InD=ind)
    elif ds_name == 'cifar100':
        dataset = CIFAR100(path='data/CIFAR100', n_class=n_class, InD=ind)
    elif ds_name == 'celeba':
        dataset = CELEBA(path='data/CELEBA', InD=ind) 
    elif ds_name == 'lsun':
        dataset = LSUN(path='data/LSUN', InD=ind)
    elif ds_name in ['random', 'const']:
        dataset = FAKE(ds_name, ind)
    elif 'gaussian' in ds_name:
        dataset = GAUSSIAN(ds_name)
    else:
        raise ValueError('Unknown dataset')
    return dataset

def postprocess_glow(x):
    # per-image 0-1 normalization
    b,c,h,w = x.shape
    x = x.view(b, -1)
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = x.view(b, c, h, w)
    x = 255 - x * 255
    return x.byte()

def postprocess_realnvp(x, c, h, w):
    # # 0-1 normalization
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = x.view(x.size(0), c, h, w)
    x = 255 - x * 255
    return x.byte()


def main():
    """
    Main Function.

    Training/Test/Plot
    """
    args = parse_arguments()
    device = torch.device('cuda')

    # remove randomness
    set_random_seed(args.seed)

    dataset = setDataset(args.dataset, args.n_class, args.ind, False, args.seg_len)

    # Set Model
    if (args.autoencoder is None):
        c, h, w = dataset.shape
        # build Density Estimator
        if args.estimator == 'REALNVP':
            if args.dataset in ['cifar10','cifar100','svhn','celeba','lsun']:
                z_shape = (64, 3072)
            elif args.dataset in ['mnist','fmnist','kmnist','omniglot']:
                z_shape = (64, 784)
            model = TinvREALNVP(args.num_blocks, c*h*w, args.hidden_size).cuda()
            if args.num_epochs == -1:
                model_path = f'./checkpoints/{args.dataset}/b{args.num_blocks}h{args.hidden_size}c{args.code_length}/{args.dataset}REALNVP.pkl'
            else:
                model_path = f'./checkpoints/{args.dataset}/b{args.num_blocks}h{args.hidden_size}c{args.code_length}/{args.dataset}REALNVP_{args.num_epochs}.pkl'
            model.load_state_dict(torch.load(model_path), strict=False)
            model = model.eval()
            z = torch.randn(z_shape).to(device)
            images, _ = model(z, mode='inverse')
            images = postprocess_realnvp(images, c, h, w).cpu()
            grid = utils.make_grid(images, nrow=8)
            utils.save_image(grid, f'./Generated_imgs_RealNVP_{args.dataset}_{args.num_epochs}_b{args.num_blocks}h{args.hidden_size}.png')
        elif args.estimator == 'GLOW':
            model = Glow(
                image_shape=(c, h, w),
                hidden_channels=args.hidden_size,
                K=args.K,
                L=args.num_blocks,
                actnorm_scale=1.0,
                flow_permutation='invconv',
                flow_coupling='affine',
                LU_decomposed=True,
                y_classes=10,
                learn_top=True,
                y_condition=False,
                ).cuda()

            if args.dataset in ['cifar10','cifar100','svhn','celeba','lsun']:
                z_shape = (64, 48, 4, 4)
            elif args.dataset in ['mnist','fmnist','kmnist','omniglot']:
                z_shape = (64, 8, 7, 7)

            if args.num_epochs == -1:
                model_path = f'./checkpoints/{args.dataset}/K{args.K}L{args.num_blocks}h{args.hidden_size}/{args.dataset}GLOW.pkl'
            else:
                model_path = f'./checkpoints/{args.dataset}/K{args.K}L{args.num_blocks}h{args.hidden_size}/{args.dataset}GLOW_{args.num_epochs}.pkl'
            model.load_state_dict(torch.load(model_path), strict=False)
            model.set_actnorm_init()
            model = model.eval()
            z = torch.randn(z_shape).to(device)
            images = model(z=z, temperature=1, reverse=True)
            images = postprocess_glow(images).cpu()
            grid = utils.make_grid(images, nrow=8)
            utils.save_image(grid, f'./Generated_imgs_Glow_{args.dataset}_{args.num_epochs}_K{args.K}L{args.num_blocks}h{args.hidden_size}.png')


def parse_arguments():
    """

    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Generating images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # use autoencoder (or not)
    parser.add_argument('--autoencoder', type=str, help='The Autoencoder framework. Choose among `LSA`', metavar='')
    # density estimator / flow model
    parser.add_argument('--estimator', type=str,
                        help='The name of density estimator / flow model. Choose among `REALNVP`', metavar='')
    # Dataset
    parser.add_argument('--dataset', type=str, help='The name of the dataset', metavar='')
    # specify the In-Distribution (dataset name or class name)
    parser.add_argument('--ind', type=str, default=None, help='In-distribution dataset (or class) name')
    # Model specification
    parser.add_argument('--num_blocks', type=int, default=1, help='number of invertible blocks (default: 5)')
    parser.add_argument('--code_length', type=int, default=64, help='length of hidden vector (default: 64)')
    parser.add_argument('--hidden_size', type=int, default=2048, help='length of hidden vector (default: 2048)')
    parser.add_argument('--K', type=int, default=3, help='number of flow steps in Glow (original default: 3)')
    parser.add_argument('--n_class', type=int, default=10, help='Number of classes used in experiments')

    parser.add_argument('--seed', type=int, default=1, help='random_seed')
    # number of training epochs (for loading partly trained models), -1 means the model at the end of training
    parser.add_argument('--num_epochs', type=int, default=-1, help='number of training epochs (default: -1)')
    parser.add_argument('--seg_len', type=int, default=10, help='length of segment of EEG/ECG')

    return parser.parse_args()


if __name__ == '__main__':
    """
    entry point.
    """

    start_t = time.time()
    main()
    print("Time cost: ", round(time.time() - start_t, 2), "s.")
