import torch
import torch.nn.functional as F
import os
import numpy as np
from torch.autograd import grad


class Losses:

    def __init__(self, conf):
        self.conf = conf

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

        proj_gt = torch.mm(gt.reshape(self.conf['BATCHSIZE'], -1), pcvec)
        proj_pred = torch.mm(pred.reshape(self.conf['BATCHSIZE'], -1), pcvec)
        pca_loss = torch.mean(
            torch.sum(
                (proj_gt - proj_pred)**2 / pcvar, dim=1
            )
        )

        # calculating residuals. by subtracting each PCA component to the original image
        gt_err = gt
        pred_err = pred
        for i in range(self.conf['PCA_COMP_NUMBER']):
            gt_err = gt_err.reshape(self.conf['BATCHSIZE'], -1) - torch.mm(torch.mm(gt.reshape(self.conf['BATCHSIZE'], -1), pcvec), torch.t(pcvec))
            pred_err = pred_err.reshape(self.conf['BATCHSIZE'], -1) - torch.mm(torch.mm(pred.reshape(self.conf['BATCHSIZE'], -1), pcvec), torch.t(pcvec))
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
        compute the Kullback Leibler distance between the predicted distribution
        and the normal N(0, I)
        :param mu: predicted mean
        :param logvar: predicted log(variance)
        :return: kl_distance
        """
        # kl_element = torch.add(torch.add(torch.add(mu.pow(2), logvar.exp()), -1), logvar.mul(-1))
        kl_element = (-2 * logvar) + torch.exp(2 * logvar) + mu.pow(2) - 1
        return torch.mean(torch.sum(kl_element * 0.5, axis=1))

    def hist_loss(self, gt, pred, w):
        """
        calculate the loss by computing the pixelwise distance between the predicted
        color image and the gran truth color image
        :param gt: original color image (AB space)
        :param pred: predicted color image
        :return: loss, weighted according to the probability of each color
        """
        gt = gt.view(-1, self.conf['IMG_W'] * self.conf['IMG_H'] * 2)
        pred = pred.view(-1, self.conf['IMG_W'] * self.conf['IMG_H'] * 2)
        recon_element = torch.sqrt(torch.sum(torch.mul(torch.add(gt, pred.mul(-1)).pow(2), w), 1))
        return recon_element.mean()  # mean on the batch

    def l2_loss(self, gt, pred):
        """
        simple L2 loss between colored image and predicted colored image without any weight
        :param gt: original colored image (AB channels)
        :param pred: predicted colored image (AB channels)
        :return: L2 loss
        """
        recon_element_l2 = torch.sqrt(torch.sum(torch.add(gt, pred.mul(-1)).pow(2), 1))
        return torch.sum(recon_element_l2).mul(1. / self.conf['BATCHSIZE'])

    def gradient_penalty(self, input, output):

        # generating a z from the net prediction
        # stddev = torch.sqrt(torch.exp(logvar))
        # eps = torch.randn(stddev.size()).normal_().cuda()
        # z_pred = torch.add(mu, torch.mul(eps, stddev)).requires_grad_()
        # z_pred = z_pred.reshape(-1, self.conf['HIDDEN_SIZE'], 1, 1).repeat(1, 1, 4, 4)

        # computing gradient penalty
        gradients = grad(outputs=output, inputs=input,
                         grad_outputs=torch.ones_like(output).cuda(),
                         retain_graph=True, create_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
        return gradient_penalty

    def cvae_loss(self, pred, gt, lossweights, mu, logvar):
        """
        this encoder loss is not forced to be normal gaussian
        :param pred: predicted color image
        :param gt: real color image
        :param lossweights: weights for colors
        :return:
        """
        gp_1 = self.gradient_penalty(gt, mu)
        gp_2 = self.gradient_penalty(gt, logvar)
        grad_penalty = (gp_1 + gp_2) / 2

        kl_loss = self.kl_loss(mu, logvar)
        recon_loss = self.hist_loss(gt, pred, lossweights)
        # recon_loss_l2 = self.l2_loss(gt, pred)
        return recon_loss, kl_loss, grad_penalty