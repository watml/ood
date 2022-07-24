import argparse
from argparse import Namespace
import os
import torch
import numpy as np
import torch.nn as nn

# datasets
from datasets import *
# models
# auto-encoder
from models.LSA_mnist import LSA_MNIST
from models.LSA_cifar10 import LSA_CIFAR10
# density estimator
from models.transform_realnvp import TinvREALNVP
from models.glow_models import Glow
# random seed
from utils import set_random_seed
from result_helpers.ood_trainer import OODTrainer
# path
from utils import create_checkpoints_dir

import time


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

def main():
    """
    Main Function.

    Training/Test/Plot
    """
    args = parse_arguments()
    device = torch.device('cuda')

    # remove randomness
    set_random_seed(args.seed)

    # Set Dataset
    dataset = setDataset(args.dataset, args.n_class, args.ind, False, args.seg_len)

    checkpoints_dir = create_checkpoints_dir(
        args.dataset,  args.num_blocks, args.hidden_size, args.code_length, args.estimator, args.hidden_size, args.K)
    print(checkpoints_dir)

    # Set Model
    if (args.autoencoder is None):
        print ('No Autoencoder')
        c, h, w = dataset.shape
        # build Density Estimator
        if args.estimator == 'REALNVP':
            model = TinvREALNVP(args.num_blocks, c*h*w, args.hidden_size).cuda()
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
        else:
            raise ValueError('Unknown Estimator')
    else:
        print(f'Autoencoder:{args.autoencoder}')
        print(f'Density Estimator:{args.estimator}')
        if args.autoencoder == "LSA":
            if args.dataset in ['mnist', 'fmnist']:
                model = LSA_MNIST(
                    input_shape=dataset.shape,
                    code_length=args.code_length,
                    num_blocks=args.num_blocks,
                    est_name=args.estimator,
                    hidden_size=args.hidden_size).cuda()
            elif args.dataset in ['cifar10', 'svhn', 'cifar100', 'celeba']:
                model = LSA_CIFAR10(
                    input_shape=dataset.shape,
                    code_length=args.code_length,
                    K=args.K,
                    num_blocks=args.num_blocks,
                    est_name=args.estimator,
                    hidden_size=args.hidden_size,
                    k=args.k,
                    r=args.r).cuda()
            else:
                ValueError("Unknown Dataset")
        else:
            raise ValueError('Unknown Autoencoder')


    trainer = OODTrainer(
        dataset=dataset,
        model=model,
        lam=args.lam,
        checkpoints_dir=checkpoints_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        code_length=args.code_length,
        log_step=args.log_step,
        device=device,
        InD=args.ind,
        num_epochs=args.num_epochs,
        noise_flag=args.noise_flag,
        sigma=args.sigma)

    if args.trainflag:
        trainer.train_ood_exp(args.ind)
    elif args.testflag:
        InD = args.ind
        if len(InD) > 1:
            trainer.test_ood_exp(args.ind)
            dataset_InD = setDataset(InD, args.n_class, InD, False, args.seg_len)
            trainer_InD = OODTrainer(
                dataset=dataset_InD,
                model=model,
                lam=args.lam,
                checkpoints_dir=checkpoints_dir,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                code_length=args.code_length,
                log_step=args.log_step,
                device=device,
                InD=args.ind,
                num_epochs=args.num_epochs,
                noise_flag=args.noise_flag,
                sigma=args.sigma)
            trainer_InD.test_ood_exp(args.ind)
            # plot histogram and ROC
            if args.density_rule_flag:
                trainer.plotDensityRule() # plot histogram and roc using density rule
            elif args.kst_rule_flag:
                if (args.autoencoder is None):
                    trainer.plotKSTRuleRandPJ(args.num_project)  # plot histogram and roc using difference between distributions rule (ks-test)
                else:
                    trainer.plotKSTRule()
            elif args.typical_flag:
                epsilon = trainer_InD.getEpsilon()
                trainer.plotTypicalityTest(epsilon)
            elif args.klod_flag:
                trainer.plotKLOD()
        else:
            trainer.test_ood_exp(args.ind)
            dataset_InD = setDataset(args.dataset, args.n_class, InD, True, args.seg_len)
            trainer_InD = OODTrainer(
                dataset=dataset_InD,
                model=model,
                lam=args.lam,
                checkpoints_dir=checkpoints_dir,
                batch_size=args.batch_size,
                lr=args.lr,
                epochs=args.epochs,
                code_length=args.code_length,
                log_step=args.log_step,
                device=device,
                InD=args.ind,
                num_epochs=args.num_epochs,
                noise_flag=args.noise_flag,
                sigma=args.sigma)
            trainer_InD.test_ood_exp(args.ind)
            if args.density_rule_flag:
                trainer.plotDensityRule() # plot histogram and roc using density rule
            elif args.kst_rule_flag:
                if (args.autoencoder is None):
                    pass
                    # trainer.plotKSTRuleRandPJ(args.num_project)  # plot histogram and roc using difference between distributions rule (ks-test)
                else:
                    trainer.plotKSTRule()
            elif args.klod_flag:
                trainer.plotKLOD()
            elif args.typical_flag:
                epsilon = trainer_InD.getEpsilon()
                trainer.plotTypicalityTest(epsilon)

def parse_arguments():
    """

    Argument parser.

    :return: the command line arguments.
    """
    parser = argparse.ArgumentParser(description='OOD detection with flow models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # use autoencoder (or not)
    parser.add_argument( '--autoencoder', type=str, help='The Autoencoder framework. Choose among `LSA`', metavar='')

    # density estimator / flow model
    parser.add_argument('--estimator', type=str, help='The name of density estimator / flow model. Choose among `REALNVP`', metavar='')

    # Dataset
    parser.add_argument('--dataset', type=str, help='The name of the dataset', metavar='')
    # specify the In-Distribution (dataset name or class name)
    parser.add_argument('--ind', type=str, default=None, help='In-distribution dataset (or class) name')

    # Setting model mode (Train or Test)
    parser.add_argument('--Train', dest='trainflag', action='store_true', default=False, help='Train Mode')
    parser.add_argument('--Test', dest='testflag', action='store_true', default=False, help='Test Mode')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate (default: 0.00001)')

    # model specifications
    parser.add_argument('--num_blocks', type=int, default=1, help='number of invertible blocks (default: 1)')
    parser.add_argument('--code_length', type=int, default=64, help='length of latent code of autoencoder (default: 64)')
    parser.add_argument('--hidden_size', type=int, default=2048, help='length of hidden vector (default: 2048)')
    parser.add_argument('--K', type=int, default=3, help='number of flow steps in Glow (original default: 32)')
    parser.add_argument('--n_class', type=int, default=10, help='Number of classes used in experiments')


    # detection rules
    parser.add_argument('--density_rule', dest='density_rule_flag', action='store_true', default=False, help='plot histogram and ROC using density as detection score')
    parser.add_argument('--kst_rule', dest='kst_rule_flag', action='store_true', default=False, help='plot histogram and ROC using discrepancy rule (ks-test)')
    parser.add_argument('--klod', dest='klod_flag', action='store_true', default=False, help='plot histogram and ROC using KLOD')
    parser.add_argument('--typical', dest='typical_flag', action='store_true', default=False, help='typicality test')

    # number of projected dimensions
    parser.add_argument('--num_project', type=int, default=200, help='number of projected dimensions (default: 200)')
    # number of epochs for loading models
    parser.add_argument('--num_epochs', type=int, default=-1, help='for loading differently trained models')
    parser.add_argument('--lam', type=float, default=1.0, help='trade off between reconstruction loss and auto-regression loss')
    parser.add_argument('--seed', type=int, default=1, help='random_seed')
    parser.add_argument('--log_step', type=int, default=100, help='log_step, save model for every #log_step epochs')

    parser.add_argument('--add_noise', dest='noise_flag', action='store_true', default=False, help='add noise to input images')
    parser.add_argument('--sigma', type=float, default=1.0, help='sigma of noise')
    parser.add_argument('--seg_len', type=int, default=10, help='length of segment of EEG/ECG')

    return parser.parse_args()


if __name__ == '__main__':
    """
    entry point.
    """

    start_t = time.time()
    main()
    print("Time cost: ", round(time.time() - start_t, 2), "s.")
