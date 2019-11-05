from vae import VAE
from mdn import MDN
from cvae import CVAE
import conf
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import utilities
from tensorboardX import SummaryWriter
import os
from torch.optim.lr_scheduler import StepLR


def model_a(load_vae=False, load_mdn=False):

    # tensorboard log
    writer = SummaryWriter()

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

    data, data_loader = utilities.load_data('train')
    nbatches = np.int_(np.floor(data.img_num / conf.BATCHSIZE))

    ##############
    # TRAINING VAE
    ##############
    if load_vae:
        vae.load_state_dict(torch.load(os.path.join(conf.OUT_DIR, 'models/model_vae.pth')))
        print("weights for vae loaded.")
    else:
        print("starting VAE Training for model A")
        vae.train(True)
        optimizer = optim.Adam(vae.parameters(), lr=conf.VAE_LR)
        scheduler = StepLR(optimizer, step_size=conf.SCHED_VAE_STEP, gamma=conf.SCHED_VAE_GAMMA)
        i = 0
        for epochs in range(conf.EPOCHS):
            for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in \
                    tqdm(enumerate(data_loader), total=nbatches):
                input_color = batch.cuda()
                lossweights = batch_weights.cuda()
                lossweights = lossweights.view(conf.BATCHSIZE, -1)

                optimizer.zero_grad()
                mu, logvar, color_out = vae(color=input_color, z_in=None)
                kl_loss, recon_loss, grad_loss, mah_loss = utilities.vae_loss(
                    mu,
                    logvar,
                    color_out,
                    input_color,
                    lossweights
                )
                loss = kl_loss.mul(conf.KL_W) + recon_loss.mul(conf.HIST_W) + grad_loss.mul(conf.GRA_W) + mah_loss.mul(conf.MAH_W)
                loss.backward()
                optimizer.step()
                writer.add_scalar('VAE_Loss', loss.item(), i)
                writer.add_scalar('VAE_Loss_grad', grad_loss, i)
                writer.add_scalar('VAE_Loss_kl', kl_loss, i)
                writer.add_scalar('VAE_Loss_mah', mah_loss, i)
                writer.add_scalar('one weight', vae.enc_fc1.weight.cpu().detach().numpy()[0][0], i)
                writer.add_scalar('one grad', vae.enc_fc1.weight.grad.cpu().detach().numpy()[0][0], i)
                writer.add_scalar('VAE_Loss_hist', recon_loss, i)
                # writer.add_histogram('weights', vae.enc_fc1.weight.cpu().detach().numpy(), i)
                i = i + 1
            torch.save(vae.state_dict(), '%s/models/model_vae.pth' % conf.OUT_DIR)
            scheduler.step()

        print("VAE training completed...")

    vae.train(False)

    ##############
    # TRAINING MDN
    ##############

    if load_mdn:
        mdn.load_state_dict(torch.load(os.path.join(conf.OUT_DIR, 'models/model_mdn.pth')))
        print("weights for mdn loaded.")
    else:
        print("starting MDN training...")
        mdn.train(True)
        optimizer = optim.Adam(mdn.parameters(), lr=conf.MDN_LR)
        scheduler = StepLR(optimizer, step_size=conf.SCHED_MDN_STEP, gamma=conf.SCHED_MDN_GAMMA)
        i = 0
        for epochs in range(conf.EPOCHS):
            for batch_idx, (batch, batch_recon_const, batch_weights, _, batch_feats) in \
                    tqdm(enumerate(data_loader), total=nbatches):

                input_color = batch.cuda()
                input_feats = batch_feats.cuda()

                optimizer.zero_grad()

                mu, logvar, _ = vae(color=input_color, z_in=None)
                mdn_gmm_params = mdn(input_feats)

                loss, loss_l2 = utilities.mdn_loss(
                    mdn_gmm_params,
                    mu,
                    torch.sqrt(torch.exp(logvar)),
                    conf.BATCHSIZE
                )
                loss.backward()
                optimizer.step()
                writer.add_scalar('MDN_Loss', loss.item(), i)
                i = i + 1

            torch.save(mdn.state_dict(), '%s/models/model_mdn.pth' % conf.OUT_DIR)
            scheduler.step()
        print("MDN training completed. starting testing.")


    data, data_loader = utilities.load_data('test')
    nbatches = np.int_(np.floor(data.img_num / conf.BATCHSIZE))

    ###################
    # TESTING VAE + MDN
    ###################


    mdn.train(False)
    for batch_idx, (batch, batch_recon_const, batch_weights,
                    batch_recon_const_outres, batch_feats) in \
            tqdm(enumerate(data_loader), total=nbatches):

        input_feats = batch_feats.cuda()  # grey image

        mdn_gmm_params = mdn(input_feats)
        gmm_mu, gmm_pi = utilities.get_gmm_coeffs(mdn_gmm_params)
        # creating indexes and ordering means according to the gaussian weights Pi
        pi_indexes = gmm_pi.argsort(dim=1)
        gmm_mu = gmm_mu.reshape(conf.BATCHSIZE, conf.NMIX, conf.HIDDENSIZE)
        for i in range(conf.BATCHSIZE):
            gmm_mu[i, ...] = gmm_mu[i, pi_indexes[i], ...]
        # calculating results foreach sample
        result = []
        for mix in range(conf.NMIX):
            _, _, color_out = vae(color=None, z_in=gmm_mu[:, mix, :])
            result.append(color_out.unsqueeze(dim=1))
        result = torch.cat(result, dim=1)  # concat over the num-mix dimension
        data.dump_results(
            color=result,  # batch of 8 predictions  for AB channels
            grey=batch_recon_const,  # batch of original grey channel
            gt=batch,  # batch of gt AB channels
            name=batch_idx,
        )


def model_b(load_cvae=False):

    ######################
    # CREATING THE NETWORK
    ######################

    print("creating model B architecture...")
    cvae = CVAE()
    cvae.cuda()

    ####################
    # LOAD TRAINING DATA
    ####################

    data, data_loader = utilities.load_data('train')
    nbatches = np.int_(np.floor(data.img_num / conf.BATCHSIZE))

    ###############
    # TRAINING CVAE
    ###############

    if load_cvae:
        cvae.load_state_dict(torch.load(os.path.join(conf.OUT_DIR, 'models/model_cvae.pth')))
        print("weights for vae loaded.")
    else:
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
                recon_loss, recon_loss_l2 = utilities.cvae_loss(
                    color_out,
                    input_color,
                    lossweights,
                )
                loss = recon_loss
                recon_loss_l2.detach()
                loss.backward()
                optimizer.step()
            torch.save(cvae.state_dict(), '%s/models/model_cvae.pth' % conf.OUT_DIR)

        print("CVAE training completed. starting test")

    ##############
    # TESTING CVAE
    ##############

    data, data_loader = utilities.load_data('test')
    nbatches = np.int_(np.floor(data.img_num / conf.BATCHSIZE))

    cvae.train(False)
    for batch_idx, (batch, batch_recon_const, batch_weights,
                    batch_recon_const_outres, batch_feats) in \
            tqdm(enumerate(data_loader), total=nbatches):

        input_feats = batch_feats.cuda()

        input_greylevel = batch_recon_const.cuda()

        color_out = cvae(color=None, greylevel=input_greylevel)

        data.dump_results(
            color=color_out,  # batch of 8 predictions  for AB channels
            grey=batch_recon_const,  # batch of original grey channel
            gt=batch,  # batch of gt AB channels
            name=batch_idx,
            nmix=1,
        )

    print("CVAE testing completed. check out the results in ", conf.OUT_DIR)