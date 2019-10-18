
import conf
import torch
import torch.nn.functional as F




def vae_loss(mu, logvar, pred, gt, lossweights, batchsize):
    # Kullback–Leibler divergence to force color encoder to the normal distribution
    kl_element = torch.add(torch.add(torch.add(mu.pow(2), logvar.exp()), -1), logvar.mul(-1))
    kl_loss = torch.sum(kl_element).mul(.5)
    # L-Hist
    gt = gt.view(-1, 64*64*2)
    pred = pred.view(-1, 64*64*2)
    recon_element = torch.sqrt(torch.sum(torch.mul(torch.add(gt, pred.mul(-1)).pow(2), lossweights), 1))
    recon_loss = torch.sum(recon_element).mul(1./(batchsize))
    # normal L2 loss
    recon_element_l2 = torch.sqrt(torch.sum(torch.add(gt, pred.mul(-1)).pow(2), 1))
    recon_loss_l2 = torch.sum(recon_element_l2).mul(1./(batchsize))

    return kl_loss, recon_loss, recon_loss_l2


def get_gmm_coeffs(gmm_params):
    gmm_mu = gmm_params[..., :conf.HIDDENSIZE*conf.NMIX]
    gmm_mu.contiguous()
    gmm_pi_activ = gmm_params[..., conf.HIDDENSIZE*conf.NMIX:]
    gmm_pi_activ.contiguous()
    gmm_pi = F.softmax(gmm_pi_activ, dim=1)
    return gmm_mu, gmm_pi


def mdn_loss(gmm_params, mu, stddev, batchsize):
    gmm_mu, gmm_pi = get_gmm_coeffs(gmm_params)
    eps = torch.randn(stddev.size()).normal_().cuda()
    z = torch.add(mu, torch.mul(eps, stddev))
    z_flat = z.repeat(1, conf.NMIX)
    z_flat = z_flat.view(batchsize*conf.NMIX, conf.HIDDENSIZE)
    gmm_mu_flat = gmm_mu.reshape(batchsize*conf.NMIX, conf.HIDDENSIZE)
    dist_all = torch.sqrt(torch.sum(torch.add(z_flat, gmm_mu_flat.mul(-1)).pow(2).mul(50), 1))
    dist_all = dist_all.view(batchsize, conf.NMIX)
    dist_min, selectids = torch.min(dist_all, 1)
    gmm_pi_min = torch.gather(gmm_pi, 1, selectids.view(-1, 1))
    gmm_loss = torch.mean(torch.add(-1*torch.log(gmm_pi_min+1e-30), dist_min))
    gmm_loss_l2 = torch.mean(dist_min)
    return gmm_loss, gmm_loss_l2