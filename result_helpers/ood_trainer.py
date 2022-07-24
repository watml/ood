"""

Train and Test

"""
import matplotlib
matplotlib.use('Agg')
from os.path import join
import os
import torch.optim as optim
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.loss_functions import *
from result_helpers.utils import *
from utils import *
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import norm
from scipy.stats import ks_2samp, kstest



def _init_fn():
    np.random.seed(12)


class OODTrainer(object):
    def __init__(
            self,
            dataset,
            model,
            lam,
            checkpoints_dir,
            batch_size,
            lr,
            epochs, # number of training epochs
            code_length,
            log_step,
            device,
            InD,
            num_epochs, # epochs for loading models (which may be partially trained)
            noise_flag=False,
            sigma=1.0):

        self.dataset = dataset
        self.model = model
        self.device = device
        self.InD = InD
        self.train_epochs = epochs
        self.num_epochs = num_epochs
        self.checkpoints_dir = checkpoints_dir
        self.log_step = log_step
        self.name = model.name
        self.batch_size = batch_size
        self.lam = lam
        self.noise_flag = noise_flag
        self.sigma = sigma
        self.code_length = code_length
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)

        ## Set up loss function
        if self.name in ['LSA_REALNVP']:
            self.loss = LSASOSLoss(lam)
        elif self.name in ['GLOW']:
            self.loss = GLOWLoss()
        elif self.name in ['REALNVP']:
            self.loss = SOSLoss()
        else:
            ValueError("Wrong Model Name")

        print(f"Testing on {self.name}")

    def get_path(self, InD):
        name = self.name
        dataset = self.dataset
        checkpoints_dir = self.checkpoints_dir

        self.model_dir = join(checkpoints_dir, f'{InD}{name}.pkl')

        self.train_llk_dir = join(checkpoints_dir, f'{InD}{self.name}_train_llk')
        self.val_llk_dir = join(checkpoints_dir, f'{InD}{self.name}_val_llk')
        if len(InD) == 1:
            self.test_llk_dir = join(checkpoints_dir, f'{InD}_{InD}_{self.name}_test_llk')
        else:
            self.test_llk_dir = join(checkpoints_dir, f'{InD}_{dataset.name}_{self.name}_test_llk')


    def _eval(self, x, average=True):

        if self.name in ['LSA_REALNVP']:
            x_r, z, s, log_jacob_T_inverse = self.model(x)
            tot_loss = self.loss(x, x_r, s, log_jacob_T_inverse, average)
        elif self.name in ['REALNVP']:
            s, log_jacob_T_inverse = self.model(x)
            tot_loss = self.loss(s, log_jacob_T_inverse, average)
        elif self.name == 'GLOW':
            s, _, _, nll = self.model(x, None)
            if s.dim() > 2:
                s = s.view(-1, s.shape[1]*s.shape[2]*s.shape[3])
            tot_loss = self.loss(nll, average)

        # s: source or latent variable of flow model
        # z: latent variable of autoencoder, so z only appears when LSA (an autoencoder) is used.
        if self.name in ['LSA_REALNVP']:
            return tot_loss, z, s
        else:
            return tot_loss, s

    def train_every_epoch(self, epoch, InD):
        epoch_loss = 0
        epoch_recloss = 0
        epoch_nllk = 0

        log_step = self.log_step

        self.dataset.train(InD)
        # self.model.train()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=_init_fn,
                            num_workers=0)
        epoch_size = self.dataset.length
        # s: source or latent variable of flow model
        # llk: log likelihood
        # nllk: negative log likelihood
        sample_s = []
        sample_llk = []

        for i, (x, y) in enumerate(loader):
            # print(x.shape)
            x = x.to(self.device)
            self.optimizer.zero_grad()

            if epoch == self.train_epochs - 1 or epoch % log_step == 0: # save the result of the last training epoch and every log_step epochs
                if self.name in ['LSA_REALNVP']:
                    _,  _, s = self._eval(x, average=False)
                    sample_llk.append((self.loss.autoregression_loss*(-1)).tolist())
                    sample_s.append(s.detach().cpu().numpy())
                elif self.name in ['GLOW','REALNVP']:
                    _, s = self._eval(x, average=False)
                    sample_llk.append((self.loss.autoregression_loss*(-1)).tolist())
                    sample_s.append(s.detach().cpu().numpy())

            self._eval(x) # default: average=True when training
            self.loss.total_loss.backward()
            self.optimizer.step()

            epoch_loss += self.loss.total_loss.item() * x.shape[0]

            if self.name in ['LSA_REALNVP']:
                epoch_recloss += self.loss.reconstruction_loss.item() * x.shape[0]
                epoch_nllk += self.loss.autoregression_loss.item() * x.shape[0]

        print('Current epoch: ',epoch, 'Total training epochs: ', self.train_epochs)
        if self.name in ['LSA_REALNVP']:
            print('{}Train Epoch-{}: {}\tLoss: {:.6f}\tRec: {:.6f}\tNllk:{:.6f}\t'.format(self.name,
                self.dataset.InD, epoch, epoch_loss / epoch_size, epoch_recloss / epoch_size,  epoch_nllk / epoch_size))
        else:
            print('Train Epoch-{}: {}\tLoss:{:.6f}\t'.format(
                self.dataset.InD, epoch, epoch_loss / epoch_size))

        # save results
        if epoch == self.train_epochs - 1 or epoch % log_step == 0:
            if epoch < self.train_epochs - 1:  # save every log_step epochs
                np.savez(f'{self.train_llk_dir}_{epoch}', llk=sample_llk, s=sample_s)
            else:  # save the log likelihood of the last training epoch
                np.savez(self.train_llk_dir, llk=sample_llk, s=sample_s)

        return epoch_loss / epoch_size, epoch_recloss / epoch_size, epoch_nllk / epoch_size

    def validate(self, epoch, InD):
        val_loss = 0
        val_nllk = 0
        val_rec = 0
        bs = self.batch_size
        log_step = self.log_step
        self.dataset.val(InD)
        loader = DataLoader(self.dataset, bs, shuffle=False)
        epoch_size = self.dataset.length
        sample_llk = []
        sample_s = []

        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(self.device)
            with torch.no_grad():
                self._eval(x, average=False)
                if epoch == self.train_epochs - 1 or epoch % log_step == 0:
                    if self.name in ['LSA_REALNVP']:
                        _, _, s = self._eval(x, average=False)
                        sample_llk.append((self.loss.autoregression_loss * (-1)).tolist())
                        sample_s.append(s.detach().cpu().numpy())
                    elif self.name in ['GLOW','REALNVP']:
                        _, s = self._eval(x, average=False)
                        sample_llk.append((self.loss.autoregression_loss * (-1)).tolist())
                        sample_s.append(s.detach().cpu().numpy())
                if self.name in ['LSA_REALNVP']:
                    val_nllk += self.loss.autoregression_loss.sum().item()
                    val_rec += self.loss.reconstruction_loss.sum().item()
                    val_loss = val_nllk + val_rec
                else:
                    val_loss += self.loss.total_loss.sum().item()

        if self.name in ['LSA_REALNVP']:
            print('Val_loss:{:.6f}\t Rec: {:.6f}\t Nllk: {:.6f}'.format(val_loss / epoch_size, val_rec / epoch_size,
                                                                        val_nllk / epoch_size))
        else:
            print('Val_loss:{:.6f}\t'.format(val_loss / epoch_size))

        # save results
        if epoch == self.train_epochs - 1 or epoch % log_step == 0:
            if epoch < self.train_epochs - 1:
                np.savez(f'{self.val_llk_dir}_{epoch}', llk=sample_llk, s=sample_s)
            else:
                np.savez(self.val_llk_dir, llk=sample_llk, s=sample_s)

        return val_loss / epoch_size, val_rec / epoch_size, val_nllk / epoch_size

    def train_ood_exp(self, InD):
        # type: () -> None
        """
        Actually performs trains.
        """
        self.get_path(InD)
        for epoch in range(self.train_epochs):
            model_dir_epoch = join(self.checkpoints_dir, f'{InD}{self.name}_{epoch}.pkl')
            # train every epoch
            self.model.train()
            self.train_every_epoch(epoch, InD)
            # validate every epoch
            self.model.eval()
            self.validate(epoch, InD)
            if (epoch % self.log_step == 0):
                torch.save(self.model.state_dict(), model_dir_epoch)

        print(">>> Training finish! In-Distribution: ", InD)
        torch.save(self.model.state_dict(), self.model_dir)

    def test_ood_exp(self, InD):
        self.get_path(InD)
        if len(InD)>1:
            checkpoints_dir = self.checkpoints_dir
            dir2 = os.path.basename(os.path.normpath(checkpoints_dir))
            self.train_result_dir = join(f'checkpoints/{InD}', dir2)
            if self.num_epochs == -1:
                self.model_dir = join(self.train_result_dir, f'{InD}{self.name}.pkl')
            else:
                self.model_dir = join(self.train_result_dir, f'{InD}{self.name}_{self.num_epochs}.pkl')
        else:
            checkpoints_dir = self.checkpoints_dir
            dir2 = os.path.basename(os.path.normpath(checkpoints_dir))
            self.train_result_dir = join(f'checkpoints/{self.dataset.name}', dir2)
            if self.num_epochs == -1:
                self.model_dir = join(self.train_result_dir, f'{InD}{self.name}.pkl')
            else:
                self.model_dir = join(self.train_result_dir, f'{InD}{self.name}_{self.num_epochs}.pkl')

        # load the checkpoint
        bs = self.batch_size
        if self.name == 'GLOW':
            self.model.load_state_dict(torch.load(self.model_dir), strict=False)
            self.model.set_actnorm_init()
        else:
            self.model.load_w(self.model_dir)
        print(f"Load Model from {self.model_dir}")

        self.model.eval()
        # Test sets
        self.dataset.test(InD)
        loader = DataLoader(self.dataset, batch_size=bs, shuffle=False)

        sample_llk = []
        sample_s = []

        for i, (x, y) in tqdm(enumerate(loader), desc=f'Computing scores for {self.dataset}'):
            if self.noise_flag:
                if self.dataset.name != self.InD: # it's OOD
                    x += torch.normal(torch.zeros(*x.shape), torch.ones(*x.shape)*self.sigma)
            x = x.to(self.device)
            with torch.no_grad():
                if self.name in ['LSA_REALNVP']:
                    tot_loss, z, s = self._eval(x, average=False)
                    sample_llk.append((self.loss.autoregression_loss * (-1)).tolist())
                    sample_s.append(s.detach().cpu().numpy())
                elif self.name in ['GLOW','REALNVP']:
                    tot_loss, s = self._eval(x, average=False)
                    sample_llk.append((self.loss.autoregression_loss * (-1)).tolist())
                    sample_s.append(s.detach().cpu().numpy())

        np.savez(self.test_llk_dir, llk=sample_llk, s=sample_s)

    def plotDensityRule(self):
        # using log density as detection score
        print('Plotting histograms and ROC using density rule...')
        InD = self.InD
        OOD = self.dataset.name
        model_name = self.name
        num_epochs = self.num_epochs

        test_ood_file_name = join(self.checkpoints_dir, f'{InD}_{OOD}_{model_name}_test_llk.npz')
        test_ind_file_name = join(self.checkpoints_dir, f'{InD}_{InD}_{model_name}_test_llk.npz')
        if num_epochs == -1:
            train_file_name = join(self.train_result_dir, f'{InD}{model_name}_train_llk.npz')
        else:
            train_file_name = join(self.train_result_dir, f'{InD}{model_name}_train_llk_{num_epochs}.npz')

        data = np.load(train_file_name, allow_pickle=True)
        llk_train = get_q_array(data['llk'])

        data = np.load(test_ind_file_name, allow_pickle=True)
        llk_test_ind = get_q_array(data['llk'])

        data = np.load(test_ood_file_name, allow_pickle=True)
        llk_test_ood = get_q_array(data['llk'])

        if model_name in ['GLOW', 'REALNVP']:
            print('Autoencoder is not used. Treating ', InD, 'as in-distribution and ', OOD, 'as out-of-distribution.')
            # sns.set_style('darkgrid')
            plt.figure(figsize=(5, 5))
            # rule out extreme values for plot
            llk_train = modify_abn(llk_train, np.quantile(llk_train, 0.01))
            llk_test_ind = modify_abn(llk_test_ind, np.quantile(llk_test_ind, 0.01))
            llk_test_ood = modify_abn(llk_test_ood, np.quantile(llk_test_ood, 0.01))
            plotHist(llk_train, llk_test_ind, llk_test_ood, InD, OOD, 'Log likelihood')
            fig_path = f'HistLLK_{model_name}_{InD}_{OOD}_{num_epochs}.png'
            plt.savefig(fig_path,dpi=300,bbox_inches='tight')
            plotROC(-llk_test_ind, -llk_test_ood) # since we use 1 for OOD, 0 for InD, so higher nllk implies OOD

    def plotKSTRule(self):
        # using discrepancy rule
        print('Plot histogram and ROC using discrepancy rule (ks-test)...')
        InD = self.InD
        OOD = self.dataset.name
        model_name = self.name
        bs = self.batch_size
        print('Treating ', InD, 'as in-distribution and ', OOD, 'as out-of-distribution.')

        train_file_name = join(self.train_result_dir, f'{InD}{model_name}_train_llk.npz')
        test_ood_file_name = join(self.checkpoints_dir, f'{InD}_{OOD}_{model_name}_test_llk.npz')
        test_ind_file_name = join(self.checkpoints_dir, f'{InD}_{InD}_{model_name}_test_llk.npz')
        data = np.load(test_ind_file_name, allow_pickle=True)
        s_test_ind = data['s'][:-1]

        data = np.load(test_ood_file_name, allow_pickle=True)
        s_test_ood = data['s'][:-1]

        data = np.load(train_file_name, allow_pickle=True)
        s_train = data['s']
        s_train = get_u_array(s_train) # size of training set * latent dim

        # partition s_train into batches with test batch size bs
        ss_train = []
        for ii in range(s_train.shape[0] // bs):
            ss_train.append(s_train[ii * bs: ii * bs + bs, :])
        s_train = ss_train

        plt.figure(figsize=(10, 10))

        latent_dim = s_test_ood[0].shape[1]
        s_train_all = get_u_array(s_train)
        score_test_ind_alldim = np.zeros((len(s_test_ind), latent_dim))  # number of batches * latent_dim
        score_test_ood_alldim = np.zeros((len(s_test_ood), latent_dim))  # number of batches * latent_dim

        plt.figure(figsize=(10, 10))
        for j in range(latent_dim):
            for i_test_ind in range(len(s_test_ind)):
                score_test_ind_alldim[i_test_ind, j] = ks_2samp(s_train_all[:, j], s_test_ind[i_test_ind][:, j])[0]
            for i_test_ood in range(len(s_test_ood)):
                score_test_ood_alldim[i_test_ood, j] = ks_2samp(s_train_all[:, j], s_test_ood[i_test_ood][:, j])[0]

        score_test_ind = np.mean(score_test_ind_alldim, axis=1)
        score_test_ood = np.mean(score_test_ood_alldim, axis=1)

        plt.subplot(221)
        plotHist2(score_test_ind, score_test_ood, InD, OOD, 'KS-test (distance to InD_train)')
        plt.subplot(223)
        plotROC(score_test_ind, score_test_ood)

        score_train_alldim = np.zeros((len(s_train), latent_dim)) # number of batches * latent_dim
        score_test_ind_alldim = np.zeros((len(s_test_ind), latent_dim))  # number of batches * latent_dim
        score_test_ood_alldim = np.zeros((len(s_test_ood), latent_dim))  # number of batches * latent_dim

        ref_dist = 'norm'

        for j in range(latent_dim):
            for i_train in range(len(s_train)):
                score_train_alldim[i_train, j] = kstest(s_train[i_train][:, j], ref_dist)[0]
            for i_test_ind in range(len(s_test_ind)):
                score_test_ind_alldim[i_test_ind, j] = kstest(s_test_ind[i_test_ind][:, j], ref_dist)[0]
            for i_test_ood in range(len(s_test_ood)):
                score_test_ood_alldim[i_test_ood, j] = kstest(s_test_ood[i_test_ood][:, j], ref_dist)[0]

        score_train = np.mean(score_train_alldim, axis=1)
        score_test_ind = np.mean(score_test_ind_alldim, axis=1)
        score_test_ood = np.mean(score_test_ood_alldim, axis=1)

        # remove nan or infinity
        score_train = score_train[np.where(np.isfinite(score_train))]
        score_test_ind = score_test_ind[np.where(np.isfinite(score_test_ind))]
        score_test_ood = score_test_ood[np.where(np.isfinite(score_test_ood))]

        plt.subplot(222)
        plotHist(score_train, score_test_ind, score_test_ood, InD, OOD, 'KS-test (distance to '+ref_dist+')')
        plt.subplot(224)
        plotROC(score_test_ind, score_test_ood)
        plt.tight_layout(w_pad=2.5, h_pad=3.0)
        fig_path = 'HistAUROC_discrepancy_KST_GAD_' + model_name + '_' + InD + '_' + OOD + '_bs' + str(bs) \
                   + '_c' + str(self.code_length) + '.png'
        plt.savefig(fig_path, dpi=300)


    def plotKSTRuleRandPJ(self, num_project):
        # using discrepancy rule
        print('Plot histogram and ROC using discrepancy rule (ks-test) with random projection...')
        InD = self.InD
        OOD = self.dataset.name
        model_name = self.name
        bs = self.batch_size
        num_epochs = self.num_epochs
        print('Treating ', InD, 'as in-distribution and ', OOD, 'as out-of-distribution.')

        test_ood_file_name = join(self.checkpoints_dir,f'{InD}_{OOD}_{model_name}_test_llk.npz')
        test_ind_file_name = join(self.checkpoints_dir, f'{InD}_{InD}_{model_name}_test_llk.npz')
        if num_epochs == -1:
            train_file_name = join(self.train_result_dir, f'{InD}{model_name}_train_llk.npz')
        else:
            train_file_name = join(self.train_result_dir, f'{InD}{model_name}_train_llk_{num_epochs}.npz')

        data = np.load(test_ind_file_name, allow_pickle=True)
        s_test_ind = data['s'][:-1]

        data = np.load(test_ood_file_name, allow_pickle=True)
        s_test_ood = data['s'][:-1]

        data = np.load(train_file_name, allow_pickle=True)
        s_train = data['s']
        s_train = get_u_array(s_train)

        # partition s_train into batches with test batch size bs
        ss_train = []
        for ii in range(s_train.shape[0] // bs):
            ss_train.append(s_train[ii * bs: ii * bs + bs, :])
        s_train = ss_train

        if len(InD) == 1:
            s_test_ind = get_u_array(s_test_ind)
            ss_test_ind = []
            for ii in range(s_test_ind.shape[0] // bs):
                ss_test_ind.append(s_test_ind[ii * bs: ii * bs + bs, :])
            s_test_ind = ss_test_ind
        plt.figure(figsize=(10, 10))

        latent_dim = s_test_ood[0].shape[1]

        ######## random projection matrix#########
        W = np.random.multivariate_normal(mean=np.zeros(latent_dim), cov=np.eye(latent_dim), size=(num_project))
        W_norm = np.linalg.norm(W, axis=1)  # L-2 norm
        W_norm = np.array([W_norm, ] * latent_dim).transpose()
        W = W / W_norm  # now norm of W = 1

        n_dim = W.shape[0]  # n_dim=num_project, the number of randomly projected dimensions, e.g. 100
        print('We are randomly projecting on ', n_dim, ' dimensions')

        # perform the random projection per batch (there might be faster implementation)
        s_train_new = []
        s_test_ind_new = []
        s_test_ood_new = []
        for i_train in range(len(s_train)):
            s_train_new.append(np.matmul(s_train[i_train], np.transpose(W)))
        for i_test_ind in range(len(s_test_ind)):
            s_test_ind_new.append(np.matmul(s_test_ind[i_test_ind], np.transpose(W)))
        for i_test_ood in range(len(s_test_ood)):
            s_test_ood_new.append(np.matmul(s_test_ood[i_test_ood], np.transpose(W)))

        score_test_ind_alldim = np.zeros((len(s_test_ind_new), n_dim))  # number of batches * n_dim
        score_test_ood_alldim = np.zeros((len(s_test_ood_new), n_dim))  # number of batches * n_dim

        s_train_all = get_u_array(s_train_new)

        ###### two-sample test: compare to training data #####
        print('>>> Begin Two-sample test ...')
        for j in range(n_dim):
            for i_test_ind in range(len(s_test_ind_new)):
                score_test_ind_alldim[i_test_ind, j] = ks_2samp(s_train_all[:, j], s_test_ind_new[i_test_ind][:, j])[0]
            for i_test_ood in range(len(s_test_ood_new)):
                score_test_ood_alldim[i_test_ood, j] = ks_2samp(s_train_all[:, j], s_test_ood_new[i_test_ood][:, j])[0]

        score_test_ind = np.mean(score_test_ind_alldim, axis=1)
        score_test_ood = np.mean(score_test_ood_alldim, axis=1)

        plt.subplot(221)
        plotHist2(score_test_ind, score_test_ood, InD, OOD, 'KS-test (distance to InD_train)')
        plt.subplot(223)
        plotROC(score_test_ind, score_test_ood)

        print('>>> Finish Two-sample test!')

        ref_dist = 'norm' # fixed to be normal
        score_train_alldim = np.zeros((len(s_train_new), n_dim))  # number of batches * n_dim
        score_test_ind_alldim = np.zeros((len(s_test_ind_new), n_dim))  # number of batches * n_dim
        score_test_ood_alldim = np.zeros((len(s_test_ood_new), n_dim))  # number of batches * n__dim

        ######## one-sample test: compared to a reference distribution #######
        print('>>> Begin One-sample test ...')
        for j in range(n_dim):
            for i_train in range(len(s_train_new)):
                score_train_alldim[i_train, j] = kstest(s_train_new[i_train][:, j], ref_dist)[0]
            for i_test_ind in range(len(s_test_ind_new)):
                score_test_ind_alldim[i_test_ind, j] = kstest(s_test_ind_new[i_test_ind][:, j], ref_dist)[0]
            for i_test_ood in range(len(s_test_ood_new)):
                score_test_ood_alldim[i_test_ood, j] = kstest(s_test_ood_new[i_test_ood][:, j], ref_dist)[0]

        score_train = np.mean(score_train_alldim, axis=1)
        score_test_ind = np.mean(score_test_ind_alldim, axis=1)
        score_test_ood = np.mean(score_test_ood_alldim, axis=1)

        # remove nan or infinity
        score_train = score_train[np.where(np.isfinite(score_train))]
        score_test_ind = score_test_ind[np.where(np.isfinite(score_test_ind))]
        score_test_ood = score_test_ood[np.where(np.isfinite(score_test_ood))]

        plt.subplot(222)
        plotHist(score_train, score_test_ind, score_test_ood, InD, OOD, 'KS-test (distance to '+ref_dist+')')
        plt.subplot(224)
        plotROC(score_test_ind, score_test_ood)
        print('>>> Finish one-sample test!')
        plt.tight_layout(w_pad=2.5, h_pad=3.0)
        if self.noise_flag:
            fig_path = 'HistAUROC_discrepancy_KST_GAD_' + model_name + '_' + InD + '_' + OOD + '_bs' + str(bs) \
                       + '_PJdim' + str(num_project) + '_' + str(num_epochs) + '_noise_'+ str(self.sigma) + '.png'
        else:
            fig_path = 'HistAUROC_discrepancy_KST_GAD_' + model_name + '_' + InD + '_' + OOD + '_bs' + str(bs) \
                   + '_PJdim' + str(num_project) + '_'+ str(num_epochs) + '.png'
        plt.savefig(fig_path, dpi=300)


    def getEpsilon(self):
        # compute epsilon as the threshold for typicality test using validation set
        InD = self.InD
        bs = self.batch_size

        epsilons = []

        train_file_name = join(self.train_result_dir, f'{InD}{self.name}_train_llk.npz')
        data = np.load(train_file_name, allow_pickle=True)
        llk_train = get_q_array(data['llk'])
        llk_train_mean = np.mean(llk_train)

        if self.name == 'GLOW':
            self.model.load_state_dict(torch.load(self.model_dir), strict=False)
            self.model.set_actnorm_init()
        else:
            self.model.load_w(self.model_dir)
        print(f"Load Model from {self.model_dir} for calculating epsilon")

        self.model.eval()
        # Validation sets
        self.dataset.val(InD)
        loader = DataLoader(self.dataset, batch_size=bs, shuffle=False)
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(self.device)
            with torch.no_grad():
                if batch_idx < 50: # get 50 epsilon
                    if self.name in ['REALNVP']:
                        _, nll = self.model(x)
                    else:
                        _, _, _, nll = self.model(x)
                    epsilon_k = np.abs(np.mean(-nll.cpu().numpy()) - llk_train_mean)
                    epsilons.append(epsilon_k)
        epsilon = np.quantile(epsilons, 0.99) # alpha=0.99
        print("epsilon: ", epsilon)
        return epsilon

    def plotTypicalityTest(self, epsilon):
        # reproduce typicality test result
        # The high-level idea is to compare the difference of test llk and training llk. Higher than the epsilon implies OOD.
        # As we calculate AUROC, so epsilon is not required
        # plot typicality test
        print('Plot histogram and ROC using typicality test...')
        InD = self.InD
        OOD = self.dataset.name
        model_name = self.name
        bs = self.batch_size
        print('Treating ', InD, 'as in-distribution and ', OOD, 'as out-of-distribution.')

        train_file_name = join(self.train_result_dir, f'{InD}{model_name}_train_llk.npz')
        test_ood_file_name = join(self.checkpoints_dir, f'{InD}_{OOD}_{model_name}_test_llk.npz')
        test_ind_file_name = join(self.checkpoints_dir, f'{InD}_{InD}_{model_name}_test_llk.npz')

        data = np.load(train_file_name, allow_pickle=True)
        llk_train = get_q_array(data['llk'])
        llk_train_mean = np.mean(llk_train)

        data = np.load(test_ind_file_name, allow_pickle=True)
        llk_test_ind = data['llk']

        data = np.load(test_ood_file_name, allow_pickle=True)
        llk_test_ood = data['llk']

        plt.figure(figsize=(5,10))
        score_test_ind = np.zeros(len(llk_test_ind))
        score_test_ood = np.zeros(len(llk_test_ood))

        for i_test_ind in range(len(llk_test_ind)):
            score_test_ind[i_test_ind] = np.abs(np.mean(llk_test_ind[i_test_ind]) - llk_train_mean)
        for i_test_ood in range(len(llk_test_ood)):
            score_test_ood[i_test_ood] = np.abs(np.mean(llk_test_ood[i_test_ood]) - llk_train_mean)

        # remove nan or infinity
        score_test_ind = score_test_ind[np.where(np.isfinite(score_test_ind))]
        score_test_ood = score_test_ood[np.where(np.isfinite(score_test_ood))]

        plt.subplot(211)
        plotHist2(score_test_ind, score_test_ood, InD, OOD, 'Typicality test')
        plt.subplot(212)
        plotROC(score_test_ind, score_test_ood)

        plt.tight_layout(w_pad=2.5, h_pad=3.0)
        fig_path = 'HistAUROC_typicality_GAD_' + model_name + '_' + InD + '_' + OOD + '_bs' + str(bs) + '.png'
        plt.savefig(fig_path, dpi=300)


    def plotKLOD(self):
        print('Plot histogram and ROC using KLOD...')
        InD = self.InD
        OOD = self.dataset.name
        model_name = self.name
        bs = self.batch_size
        print('Treating ', InD, 'as in-distribution and ', OOD, 'as out-of-distribution.')

        train_file_name = join(self.train_result_dir, f'{InD}{model_name}_train_llk.npz')
        test_ood_file_name = join(self.checkpoints_dir, f'{InD}_{OOD}_{model_name}_test_llk.npz')
        test_ind_file_name = join(self.checkpoints_dir,
                                 f'{InD}_{InD}_{model_name}_test_llk.npz')
        data = np.load(test_ind_file_name, allow_pickle=True)
        s_test_ind = data['s'][:-1]

        data = np.load(test_ood_file_name, allow_pickle=True)
        s_test_ood = data['s'][:-1]

        data = np.load(train_file_name, allow_pickle=True)
        s_train = data['s']
        s_train = get_u_array(s_train)

        ss_train = []
        for ii in range(s_train.shape[0] // bs):
            ss_train.append(s_train[ii * bs: ii * bs + bs, :])
        s_train = ss_train
        plt.figure(figsize=(5, 10))
        n = s_test_ood[0].shape[1]
        score_train = np.zeros(len(s_train)) # number of batches
        score_test_ind = np.zeros(len(s_test_ind))  # number of batches
        score_test_ood = np.zeros(len(s_test_ood))  # number of batches

        for i_train in range(len(s_train)):
            mean_train = np.mean(s_train[i_train], axis=0)
            cov_train = np.cov(s_train[i_train], rowvar=0)
            # print(np.linalg.det(cov_train))  # all zero
            # if i_train == 0:
            #     print(cov_train)
            #
            # temp = s_train[i_train] - np.array([mean_train,] * bs)
            # cov_train = 1.0/bs * np.matmul(np.transpose(temp), temp)
            # print(np.linalg.det(cov_train))
            # if i_train == 0:
            #     print(cov_train)
            score_train[i_train] = KLD(mean_train, cov_train, n)
        for i_test_ind in range(len(s_test_ind)):
            mean_test_ind = np.mean(s_test_ind[i_test_ind], axis=0)
            cov_test_ind = np.cov(s_test_ind[i_test_ind], rowvar=0)
            # print(np.linalg.det(cov_test_ind)) # all zero
            score_test_ind[i_test_ind] = KLD(mean_test_ind, cov_test_ind, n)
        for i_test_ood in range(len(s_test_ood)):
            mean_test_ood = np.mean(s_test_ood[i_test_ood], axis=0)
            cov_test_ood = np.cov(s_test_ood[i_test_ood], rowvar=0)
            # print(np.linalg.det(cov_test_ood)) # all zero
            score_test_ood[i_test_ood] = KLD(mean_test_ood, cov_test_ood, n)

        # remove nan or infinity
        score_train = score_train[np.where(np.isfinite(score_train))]
        score_test_ind = score_test_ind[np.where(np.isfinite(score_test_ind))]
        score_test_ood = score_test_ood[np.where(np.isfinite(score_test_ood))]

        plt.subplot(211)
        plotHist(score_train, score_test_ind, score_test_ood, InD, OOD, 'KLOD')
        plt.subplot(212)
        plotROC(score_test_ind, score_test_ood)
        plt.tight_layout(w_pad=2.5, h_pad=3.0)
        fig_path = 'HistAUROC_discrepancy_KLOD_GAD_' + model_name + '_' + InD + '_' + OOD + '_bs' + str(bs)  + '.png'
        plt.savefig(fig_path, dpi=300)



