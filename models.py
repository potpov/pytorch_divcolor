from vae import VAE
from mdn import MDN
from cvae import CVAE
import conf
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import Utils


def model_a():

    ######################
    # CREATING THE NETWORK
    ######################

    print("creating model A architecture...")
    vae = VAE()
    vae.cuda()
    mdn = MDN()
    mdn.cuda()

    ####################
    # LOAD TRAINING DATA
    ####################

    data, data_loader = Utils.load_data('train')
    nbatches = np.int_(np.floor(data.img_num / conf.BATCHSIZE))

    ##############
    # TRAINING VAE
    ##############

    print("starting VAE Training for model A")
    vae.train(True)
    optimizer = optim.Adam(vae.parameters(), lr=conf.VAE_LR)

    for epochs in range(conf.EPOCHS):
        for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in \
                tqdm(enumerate(data_loader), total=nbatches):
            input_color = batch.cuda()
            lossweights = batch_weights.cuda()
            lossweights = lossweights.view(conf.BATCHSIZE, -1)

            optimizer.zero_grad()
            mu, logvar, color_out = vae(color=input_color, z_in=None)
            # fancy LOSS Calculation
            kl_loss, recon_loss, recon_loss_l2 = Utils.vae_loss(
                mu,
                logvar,
                color_out,
                input_color,
                lossweights,
                conf.BATCHSIZE
            )
            loss = kl_loss.mul(1e-2) + recon_loss
            recon_loss_l2.detach()
            loss.backward()
            optimizer.step()
        torch.save(vae.state_dict(), '%s/models/model_vae.pth' % conf.OUT_DIR)

    ##############
    # TRAINING MDN
    ##############

    print("VAE training completed. starting MDN training...")
    vae.train(False)
    mdn.train(True)
    optimizer = optim.Adam(mdn.parameters(), lr=conf.MDN_LR)
    for epochs in range(conf.EPOCHS):
        for batch_idx, (batch, batch_recon_const, batch_weights, _, batch_feats) in \
                tqdm(enumerate(data_loader), total=nbatches):

            input_color = batch.cuda()
            input_feats = batch_feats.cuda()

            optimizer.zero_grad()

            mu, logvar, _ = vae(color=input_color, z_in=None)
            mdn_gmm_params = mdn(input_feats)

            loss, loss_l2 = Utils.mdn_loss(mdn_gmm_params, mu, torch.sqrt(torch.exp(logvar)), conf.BATCHSIZE)
            loss.backward()

            optimizer.step()

        torch.save(mdn.state_dict(), '%s/models/model_mdn.pth' % conf.OUT_DIR)
    print("MDN training completed. starting testing.")

    ###################
    # LOAD TESTING DATA
    ###################

    data, data_loader = Utils.load_data('test')
    nbatches = np.int_(np.floor(data.img_num / conf.BATCHSIZE))

    #########
    # TESTING
    #########

    mdn.train(False)
    for batch_idx, (batch, batch_recon_const, batch_weights,
                    batch_recon_const_outres, batch_feats) in \
            tqdm(enumerate(data_loader), total=nbatches):

        input_feats = batch_feats.cuda()

        mdn_gmm_params = mdn(input_feats)
        gmm_mu, gmm_pi = Utils.get_gmm_coeffs(mdn_gmm_params)
        gmm_pi = gmm_pi.view(-1, 1)
        gmm_mu = gmm_mu.reshape(-1, conf.HIDDENSIZE)

        for j in range(conf.BATCHSIZE):
            batch_j = np.tile(batch[j, ...].numpy(), (conf.BATCHSIZE, 1, 1, 1))
            batch_recon_const_j = np.tile(batch_recon_const[j, ...].numpy(), (conf.BATCHSIZE, 1, 1, 1))
            batch_recon_const_outres_j = np.tile(batch_recon_const_outres[j, ...].numpy(),
                                                 (conf.BATCHSIZE, 1, 1, 1))

            input_color = torch.from_numpy(batch_j).cuda()
            input_greylevel = torch.from_numpy(batch_recon_const_j).cuda()

            curr_mu = gmm_mu[j*conf.NMIX:(j+1)*conf.NMIX, :]
            orderid = np.argsort(
                gmm_pi[j*conf.NMIX:(j+1)*conf.NMIX, 0].cpu().data.numpy().reshape(-1))

            z = curr_mu.repeat(np.int((conf.BATCHSIZE*1.)/conf.NMIX), 1)

            _, _, color_out = vae(color=None, z_in=z)

            data.saveoutput_gt(color_out.cpu().data.numpy()[orderid, ...],
                               batch_j[orderid, ...],
                               'divcolor_%05d_%05d' % (batch_idx, j),
                               conf.NMIX,
                               net_recon_const=batch_recon_const_outres_j[orderid, ...])

    print("VAE + MDN testing completed. check out the results in ", conf.OUT_DIR)


def model_b():

    ######################
    # CREATING THE NETWORK
    ######################

    print("creating model B architecture...")
    cvae = CVAE()
    cvae.cuda()

    ####################
    # LOAD TRAINING DATA
    ####################

    data, data_loader = Utils.load_data('train')
    nbatches = np.int_(np.floor(data.img_num / conf.BATCHSIZE))

    ##############
    # TRAINING VAE
    ##############

    print("starting CVAE Training for model B")
    cvae.train(True)
    optimizer = optim.Adam(cvae.parameters(), lr=conf.VAE_LR)

    for epochs in range(conf.EPOCHS):
        for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in \
                tqdm(enumerate(data_loader), total=nbatches):

            input_color = batch.cuda()
            lossweights = batch_weights.cuda()
            lossweights = lossweights.view(conf.BATCHSIZE, -1)
            input_greylevel = batch_recon_const.cuda()

            optimizer.zero_grad()
            color_out = cvae(color=input_color, greylevel=input_greylevel)
            # fancy LOSS Calculation
            recon_loss, recon_loss_l2 = Utils.cvae_loss(
                color_out,
                input_color,
                lossweights,
                conf.BATCHSIZE
            )
            loss = recon_loss
            recon_loss_l2.detach()
            loss.backward()
            optimizer.step()
        torch.save(cvae.state_dict(), '%s/models/model_cvae.pth' % conf.OUT_DIR)

    print("CVAE training completed. starting test")

    ###################
    # LOAD TESTING DATA
    ###################

    data, data_loader = Utils.load_data('test')
    nbatches = np.int_(np.floor(data.img_num / conf.BATCHSIZE))

    #########
    # TESTING
    #########

    cvae.train(False)
    for batch_idx, (batch, batch_recon_const, batch_weights,
                    batch_recon_const_outres, batch_feats) in \
            tqdm(enumerate(data_loader), total=nbatches):

        input_feats = batch_feats.cuda()

        input_greylevel = batch_recon_const.cuda()

        color_out = cvae(color=None, greylevel=input_greylevel)

        data.saveoutput_gt(
            color_out.cpu().data.numpy(),
            batch.numpy(),  # grand truth in batch
            'divcolor_%05d' % (batch_idx),  # output name
            conf.BATCHSIZE,  # batch size (set to one because we are already looping over the batch)
            5,  # number of column
            batch_recon_const_outres.numpy()  # black and white gt
        )

    print("CVAE testing completed. check out the results in ", conf.OUT_DIR)