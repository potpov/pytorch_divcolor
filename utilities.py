from conf import default_conf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from colordata import Colordata
import os
import numpy as np
import datetime
import json


class Utilities:

    def __init__(self, load_dir=None):
        """
        loading config file from json file or
        creating a new experiment from the current default configuration
        default configuration is stored in conf.py
        :param load_dir: name of the folder in the experiment dir
        :return: new experiment dir or loaded experiment dir
        """
        if not load_dir:
            # CREATING NEW EXPERIMENT
            # generating unique name for the experiment folder
            datalog = str(datetime.datetime.now()).replace(' ', '_')
            save_dir = os.path.join(default_conf['OUT_DIR'], datalog)

            # creating folders for model, config and results
            os.mkdir(save_dir)
            if default_conf['TEST_MDN_VAE']:
                os.mkdir(os.path.join(save_dir, 'results_mdn'))
            if default_conf['TEST_CVAE']:
                os.mkdir(os.path.join(save_dir, 'results_cvae'))

            # dump configuration file
            with open(os.path.join(save_dir, 'config.json'), "w") as write_file:
                json.dump(default_conf, write_file, indent=4)
            self.save_dir = save_dir
            self.conf = default_conf

        else:
            # LOADING PREVIOUS EXPERIMENT
            with open(os.path.join(default_conf['OUT_DIR'], load_dir, 'config.json'), 'r') as handle:
                config = json.load(handle)
            self.save_dir = os.path.join(default_conf['OUT_DIR'], load_dir)
            # create results folder if does not exists
            if config['TEST_MDN_VAE'] and not os.path.isdir(os.path.join(self.save_dir, 'results_mdn')):
                os.mkdir(os.path.join(self.save_dir, 'results_mdn'))
            if config['TEST_CVAE'] and not os.path.isdir(os.path.join(self.save_dir, 'results_cvae')):
                os.mkdir(os.path.join(self.save_dir, 'results_cvae'))

            self.conf = config

    def mah_loss(self, gt, pred):
        """
        load pca vals and variance from file
        :param gt: original color space
        :param pred: predicted color space
        :return: Mahalanobis distance as described in the paper
        """
        np_pcvec = np.transpose(np.load(os.path.join(self.conf['PCA_DIR'], 'components.mat.npy')))
        np_pcvar = 1. / np.load(os.path.join(self.conf['PCA_DIR'], 'exp_variance.mat.npy'))
        pcvec = torch.from_numpy(np_pcvec[:, :self.conf['PCA_COMP_NUMBER']]).cuda()
        pcvar = torch.from_numpy(np_pcvar[:self.conf['PCA_COMP_NUMBER']]).cuda()

        proj_gt = torch.mm(gt.reshape(32, -1), pcvec)
        proj_pred = torch.mm(pred.reshape(32, -1), pcvec)
        pca_loss = torch.mean(
            torch.sum(
                (proj_gt - proj_pred)**2 / pcvar, dim=1
            )
        )

        # calculating residuals. by subtracting each PCA component to the original image
        gt_err = gt
        pred_err = pred
        for i in range(self.conf['PCA_COMP_NUMBER']):
            gt_err = gt_err.reshape(32, -1) - torch.mm(torch.mm(gt.reshape(32, -1), pcvec), torch.t(pcvec))
            pred_err = pred_err.reshape(32, -1) - torch.mm(torch.mm(pred.reshape(32, -1), pcvec), torch.t(pcvec))
        res_loss = torch.mean(
            torch.sum(
                (gt_err - pred_err) ** 2 / (pcvar[self.conf['PCA_COMP_NUMBER'] - 1] ** 2), dim=1
            )
        )
        return pca_loss + res_loss

    def grad_loss(self, gt, pred):
        # Horinzontal Sobel filter
        Sx = torch.Tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]]).cuda()
        # reshape the filter and compute the conv
        Sx = Sx.view((1, 1, 3, 3)).repeat(1, 2, 1, 1)
        G_x = F.conv2d(gt, Sx, padding=1)

        # Vertical Sobel filter
        Sy = torch.Tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]]).cuda()
        # reshape the filter and compute the conv
        Sy = Sy.view((1, 1, 3, 3)).repeat(1, 2, 1, 1)
        G_y = F.conv2d(pred, Sy, padding=1)
        G = torch.pow(G_x, 2) + torch.pow(G_y, 2)
        return torch.mean(G)

    def kl_loss(self, mu, logvar):
        """
        calculate the Kullback–Leibler distance between the predicted distribution
        and the normal N(0, I)
        :param mu: predicted mean
        :param logvar: predicted log(variance)
        :return: kl distance
        """
        kl_element = torch.add(torch.add(torch.add(mu.pow(2), logvar.exp()), -1), logvar.mul(-1))
        return torch.sum(kl_element).mul(.5)

    def hist_loss(self, gt, pred, w):
        """
        calculate the loss by computing the pixelwise distance between the predicted
        color image and the gran truth color image
        :param gt: original color image (AB space)
        :param pred: predicted color image
        :return: loss, weighted according to the probability of each color
        """
        gt = gt.view(-1, 64 * 64 * 2)
        pred = pred.view(-1, 64 * 64 * 2)
        recon_element = torch.sqrt(torch.sum(torch.mul(torch.add(gt, pred.mul(-1)).pow(2), w), 1))
        return torch.sum(recon_element).mul(1. / self.conf['BATCHSIZE'])

    def l2_loss(self, gt, pred):
        """
        simple L2 loss between colored image and predicted colored image without any weight
        :param gt: original colored image (AB channels)
        :param pred: predicted colored image (AB channels)
        :return: L2 loss
        """
        recon_element_l2 = torch.sqrt(torch.sum(torch.add(gt, pred.mul(-1)).pow(2), 1))
        return torch.sum(recon_element_l2).mul(1. / self.conf['BATCHSIZE'])

    def vae_loss(self, mu, logvar, pred, gt, lossweights):
        """
        loss for the variational autoencoder
        :param mu: predicted mean
        :param logvar: predicted logarithm of the variance
        :param pred: predicted color image
        :param gt: real color image
        :param lossweights: weight of the colors
        :return: sum of losses
        """
        kl = self.kl_loss(mu, logvar)
        recon_loss = self.hist_loss(gt, pred, lossweights)
        # recon_loss_l2 = l2_loss(gt, pred)
        mah = self.mah_loss(gt, pred)
        grad = self.grad_loss(gt, pred)
        return kl, recon_loss, grad, mah

    def cvae_loss(self, pred, gt, lossweights):
        """
        this encoder loss is not forced to be normal gaussian
        :param pred: predicted color image
        :param gt: real color image
        :param lossweights: weights for colors
        :return:
        """
        recon_loss = self.hist_loss(gt, pred, lossweights)
        recon_loss_l2 = self.l2_loss(gt, pred)
        return recon_loss, recon_loss_l2

    def get_gmm_coeffs(self, gmm_params):
        """
        return a set of means and weights for a mixture of gaussians
        :param gmm_params: predicted embedding
        :return: set of means and weights
        """
        gmm_mu = gmm_params[..., :self.conf['HIDDENSIZE'] * self.conf['NMIX']]
        gmm_mu.contiguous()
        gmm_pi_activ = gmm_params[..., self.conf['HIDDENSIZE'] * self.conf['NMIX']:]
        gmm_pi_activ.contiguous()
        gmm_pi = F.softmax(gmm_pi_activ, dim=1)
        return gmm_mu, gmm_pi


    def mdn_loss(self, gmm_params, mu, stddev, batchsize):
        gmm_mu, gmm_pi = self.get_gmm_coeffs(gmm_params)
        eps = torch.randn(stddev.size()).normal_().cuda()
        z = torch.add(mu, torch.mul(eps, stddev))
        z_flat = z.repeat(1, self.conf['NMIX'])
        z_flat = z_flat.view(batchsize * self.conf['NMIX'], self.conf['HIDDENSIZE'])
        gmm_mu_flat = gmm_mu.reshape(batchsize * self.conf['NMIX'], self.conf['HIDDENSIZE'])
        dist_all = torch.sqrt(torch.sum(torch.add(z_flat, gmm_mu_flat.mul(-1)).pow(2).mul(50), 1))
        dist_all = dist_all.view(batchsize, self.conf['NMIX'])
        dist_min, selectids = torch.min(dist_all, 1)
        gmm_pi_min = torch.gather(gmm_pi, 1, selectids.view(-1, 1))
        gmm_loss = torch.mean(torch.add(-1*torch.log(gmm_pi_min+1e-30), dist_min))
        gmm_loss_l2 = torch.mean(dist_min)
        return gmm_loss, gmm_loss_l2

    def load_data(self, dataset_type, outdir):
        data = Colordata(
            outdir,
            self.conf,
            listdir=self.conf['LISTDIR'],
            split=dataset_type
        )

        data_loader = DataLoader(
            dataset=data,
            num_workers=self.conf['NTHREADS'],
            batch_size=self.conf['BATCHSIZE'],
            shuffle=True,
            drop_last=True
        )
        return data, data_loader

