from __future__ import print_function

import argparse
import os
import numpy as np 

from colordata import colordata
from vae import VAE
from cvae import CVAE
from mdn import MDN
from logger import Logger
import conf
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import models

from tqdm import tqdm 


parser = argparse.ArgumentParser(description='PyTorch Diverse Colorization')

# parser.add_argument('dataset_key', help='Dataset')

parser.add_argument('-g', '--gpu', type=int, default=0,
                    help='gpu device id')

parser.add_argument('-lg', '--logstep', type=int, default=100,
                    help='Interval to log data')

parser.add_argument('-v', '--visdom', action='store_true',
                    help='Visdom visualization')

parser.add_argument('-s', '--server', type=str, default='http://vision-gpu-4.cs.illinois.edu',
                    help='Visdom server')

parser.add_argument('-p', '--port_num', type=int, default=8097)


args = parser.parse_args()


if(args.visdom):
  import visdom


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
    gmm_mu = gmm_params[..., :args.hiddensize*args.nmix]
    gmm_mu.contiguous()
    gmm_pi_activ = gmm_params[..., args.hiddensize*args.nmix:]
    gmm_pi_activ.contiguous()
    gmm_pi = F.softmax(gmm_pi_activ, dim=1)
    return gmm_mu, gmm_pi


def mdn_loss(gmm_params, mu, stddev, batchsize):
    gmm_mu, gmm_pi = get_gmm_coeffs(gmm_params)
    eps = torch.randn(stddev.size()).normal_().cuda()
    z = torch.add(mu, torch.mul(eps, stddev))
    z_flat = z.repeat(1, args.nmix)
    z_flat = z_flat.view(batchsize*args.nmix, args.hiddensize)
    gmm_mu_flat = gmm_mu.reshape(batchsize*args.nmix, args.hiddensize)
    dist_all = torch.sqrt(torch.sum(torch.add(z_flat, gmm_mu_flat.mul(-1)).pow(2).mul(50), 1))
    dist_all = dist_all.view(batchsize, args.nmix)
    dist_min, selectids = torch.min(dist_all, 1)
    gmm_pi_min = torch.gather(gmm_pi, 1, selectids.view(-1, 1))
    gmm_loss = torch.mean(torch.add(-1*torch.log(gmm_pi_min+1e-30), dist_min))
    gmm_loss_l2 = torch.mean(dist_min)
    return gmm_loss, gmm_loss_l2


def test_vae(model):

    model.train(False)

    batchsize = args.batchsize
    hiddensize = args.hiddensize
    nmix = args.nmix

    data = colordata(
      os.path.join(conf.OUT_DIR, 'images'),
      listdir=conf.LISTDIR,
      featslistdir=conf.FEATSLISTDIR,
      split='test')

    nbatches = np.int_(np.floor(data.img_num/batchsize))

    data_loader = DataLoader(
      dataset=data,
      num_workers=args.nthreads,
      batch_size=batchsize,
      shuffle=False,
      drop_last=True
    )
    
    test_loss = 0.
    for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in \
        tqdm(enumerate(data_loader), total=nbatches):

        input_color = batch.cuda()
        lossweights = batch_weights.cuda()
        lossweights = lossweights.view(batchsize, -1)
        input_greylevel = batch_recon_const.cuda()
        z = torch.randn(batchsize, hiddensize)

        mu, logvar, color_out = model(input_color, input_greylevel, z)
        _, _, recon_loss_l2 = \
          vae_loss(mu, logvar, color_out, input_color, lossweights, batchsize)
        test_loss = test_loss + float(recon_loss_l2.data)

    test_loss = (test_loss*1.)/nbatches

    model.train(True)
    return test_loss


def train_vae(logger=None):
    batchsize = args.batchsize
    hiddensize = args.hiddensize
    nmix = args.nmix
    nepochs = args.epochs

    data = colordata(
        os.path.join(conf.OUT_DIR, 'images'),
        listdir=conf.LISTDIR,
        featslistdir=conf.FEATSLISTDIR,
        split='train')

    nbatches = np.int_(np.floor(data.img_num/batchsize))

    data_loader = DataLoader(dataset=data, num_workers=args.nthreads,
                             batch_size=batchsize, shuffle=True, drop_last=True)

    model = CVAE()
    model.cuda()
    model.train(True)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    itr_idx = 0
    for epochs in range(nepochs):
        train_loss = 0.

        for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in \
                tqdm(enumerate(data_loader), total=nbatches):

            input_color = batch.cuda()
            lossweights = batch_weights.cuda()
            lossweights = lossweights.view(batchsize, -1)
            input_greylevel = batch_recon_const.cuda()
            z = torch.randn(batchsize, hiddensize)

            optimizer.zero_grad()
            mu, logvar, color_out = model(input_color, input_greylevel, z)

            kl_loss, recon_loss, recon_loss_l2 = \
                vae_loss(mu, logvar, color_out, input_color, lossweights, batchsize)
            loss = kl_loss.mul(1e-2)+recon_loss
            recon_loss_l2.detach()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + float(recon_loss_l2.data)

            if(logger):
                logger.update_plot(itr_idx,
                                   [kl_loss.data[0], recon_loss.data[0], recon_loss_l2.data[0]],
                                   plot_type='vae')
                itr_idx += 1

            if(batch_idx % args.logstep == 0):
                data.saveoutput_gt(color_out.cpu().data.numpy(),
                                   batch.numpy(),
                                   'train_%05d_%05d' % (epochs, batch_idx),
                                   batchsize,
                                   net_recon_const=batch_recon_const_outres.numpy())

        train_loss = (train_loss*1.)/(nbatches)
        print('[DEBUG] VAE Train Loss, epoch %d has loss %f' % (epochs, train_loss))

        test_loss = test_vae(model)
        if(logger):
            logger.update_test_plot(epochs, test_loss)
        print('[DEBUG] VAE Test Loss, epoch %d has loss %f' % (epochs, test_loss))

        torch.save(model.state_dict(), '%s/models/model_vae.pth' % (conf.OUT_DIR))


def train_mdn(logger=None):
    batchsize = args.batchsize
    hiddensize = args.hiddensize
    nmix = args.nmix
    nepochs = args.epochs_mdn

    data = colordata(
        os.path.join(conf.OUT_DIR, 'images'),
        listdir=conf.LISTDIR,
        featslistdir=conf.FEATSLISTDIR,
        split='train')

    nbatches = np.int_(np.floor(data.img_num/batchsize))

    data_loader = DataLoader(dataset=data, num_workers=args.nthreads,
                             batch_size=batchsize, shuffle=True, drop_last=True)

    model_vae = VAE()
    model_vae.cuda()
    model_vae.load_state_dict(torch.load('%s/models/model_vae.pth' % (conf.OUT_DIR)))
    model_vae.train(False)

    model_mdn = MDN()
    model_mdn.cuda()
    model_mdn.train(True)

    optimizer = optim.Adam(model_mdn.parameters(), lr=1e-3)

    itr_idx = 0
    for epochs_mdn in range(nepochs):
        train_loss = 0.

        for batch_idx, (batch, batch_recon_const, batch_weights, _, batch_feats) in \
                tqdm(enumerate(data_loader), total=nbatches):

            input_color = batch.cuda()
            input_greylevel = batch_recon_const.cuda()
            input_feats = batch_feats.cuda()
            z = torch.randn(batchsize, hiddensize)

            optimizer.zero_grad()
            # loss, loss_l2 = mdn_loss(mdn_gmm_params, mu, torch.sqrt(torch.exp(logvar)), batchsize)

            mu, logvar, _ = model_vae(input_color, input_greylevel, z)
            mdn_gmm_params = model_mdn(input_feats)

            loss, loss_l2 = mdn_loss(mdn_gmm_params, mu, torch.sqrt(torch.exp(logvar)), batchsize)
            loss.backward()

            optimizer.step()

            train_loss = train_loss + float(loss.data)

            if(logger):
                logger.update_plot(itr_idx, [loss.data[0], loss_l2.data[0]], plot_type='mdn')
                itr_idx += 1

        train_loss = (train_loss*1.)/(nbatches)
        print('[DEBUG] Training MDN, epoch %d has loss %f' % (epochs_mdn, train_loss))
        torch.save(model_mdn.state_dict(), '%s/models/model_mdn.pth' % (conf.OUT_DIR))


def divcolor():
    batchsize = args.batchsize
    hiddensize = args.hiddensize
    nmix = args.nmix

    data = colordata(
        os.path.join(conf.OUT_DIR, 'images'),
        listdir=conf.LISTDIR,
        featslistdir=conf.FEATSLISTDIR,
        split='test')

    nbatches = np.int_(np.floor(data.img_num/batchsize))

    data_loader = DataLoader(dataset=data, num_workers=args.nthreads,
                             batch_size=batchsize, shuffle=True, drop_last=True)

    model_vae = VAE()
    model_vae.cuda()
    model_vae.load_state_dict(torch.load('%s/models/model_vae.pth' % (conf.OUT_DIR)))
    model_vae.train(False)

    model_mdn = MDN()
    model_mdn.cuda()
    model_mdn.load_state_dict(torch.load('%s/models/model_mdn.pth' % (conf.OUT_DIR)))
    model_mdn.train(False)

    for batch_idx, (batch, batch_recon_const, batch_weights,
                    batch_recon_const_outres, batch_feats) in \
            tqdm(enumerate(data_loader), total=nbatches):

        input_feats = batch_feats.cuda()

        mdn_gmm_params = model_mdn(input_feats)
        gmm_mu, gmm_pi = get_gmm_coeffs(mdn_gmm_params)
        gmm_pi = gmm_pi.view(-1, 1)
        gmm_mu = gmm_mu.reshape(-1, hiddensize)

        for j in range(batchsize):
            batch_j = np.tile(batch[j, ...].numpy(), (batchsize, 1, 1, 1))
            batch_recon_const_j = np.tile(batch_recon_const[j, ...].numpy(), (batchsize, 1, 1, 1))
            batch_recon_const_outres_j = np.tile(batch_recon_const_outres[j, ...].numpy(),
                                                 (batchsize, 1, 1, 1))

            input_color = torch.from_numpy(batch_j).cuda()
            input_greylevel = torch.from_numpy(batch_recon_const_j).cuda()

            curr_mu = gmm_mu[j*nmix:(j+1)*nmix, :]
            orderid = np.argsort(
                gmm_pi[j*nmix:(j+1)*nmix, 0].cpu().data.numpy().reshape(-1))

            z = curr_mu.repeat(np.int((batchsize*1.)/nmix), 1)

            _, _, color_out = model_vae(input_color, input_greylevel, z, is_train=False)

            data.saveoutput_gt(color_out.cpu().data.numpy()[orderid, ...],
                               batch_j[orderid, ...],
                               'divcolor_%05d_%05d' % (batch_idx, j),
                               nmix,
                               net_recon_const=batch_recon_const_outres_j[orderid, ...])


if __name__ == '__main__':

    logger = None
    if(args.visdom):
        logger = Logger(args.server, args.port_num, conf.OUT_DIR)

    models.model_a()

    # train_vae(logger=logger)
    # train_mdn(logger=logger)
    # divcolor()
