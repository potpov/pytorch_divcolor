from vae import VAE
from mdn import MDN
from cvae import CVAE
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from utilities import Utilities
from tensorboardX import SummaryWriter
import os
from torch.optim.lr_scheduler import StepLR


def model(utilities):

    # tensor-board log
    writer = SummaryWriter()

    # load config file from previous or new experiments
    save_dir = utilities.save_dir
    conf = utilities.conf

    ####################
    #    VAE + MDN     #
    ####################

    print("creating model A architecture...")
    vae = VAE(conf)
    vae.cuda()
    mdn = MDN(conf)
    mdn.cuda()

    # LOAD TRAINING DATA
    data, data_loader = utilities.load_data('train', outdir=os.path.join(save_dir, 'results_mdn'))
    nbatches = np.int_(np.floor(data.img_num / conf['BATCHSIZE']))

    ##############
    # TRAINING VAE
    ##############

    if conf['LOAD_VAE']:
        print("loading vae weights.")
        vae.load_state_dict(torch.load(os.path.join(save_dir, 'model_vae.pth')))

    if conf['TRAIN_VAE']:
        print("starting VAE Training for model A")
        vae.train(True)
        optimizer = optim.Adam(vae.parameters(), lr=conf['VAE_LR'])
        scheduler = StepLR(optimizer, step_size=conf['SCHED_VAE_STEP'], gamma=conf['SCHED_VAE_GAMMA'])
        i = 0
        for epochs in range(conf['EPOCHS']):
            for batch_idx, (input_color, _, weights, _, _) in tqdm(enumerate(data_loader), total=nbatches):

                # moving to cuda
                input_color = input_color.cuda()
                lossweights = weights.cuda()
                lossweights = lossweights.view(conf['BATCHSIZE'], -1)

                # model forward
                optimizer.zero_grad()
                mu, logvar, color_out = vae(color=input_color, z_in=None)

                # computing losses
                kl_loss, recon_loss, grad_loss, mah_loss = utilities.vae_loss(
                    mu,
                    logvar,
                    color_out,
                    input_color,
                    lossweights
                )
                # summing losses
                loss = sum([
                    kl_loss.mul(conf['KL_W']),
                    recon_loss.mul(conf['HIST_W']),
                    grad_loss.mul(conf['GRA_W']),
                    mah_loss.mul(conf['MAH_W'])
                    ]
                )
                loss.backward()
                optimizer.step()

                # TENSOR BOARD DEBUG
                writer.add_scalar('VAE_Loss', loss.item(), i)
                writer.add_scalar('VAE_Loss_grad', grad_loss, i)
                writer.add_scalar('VAE_Loss_kl', kl_loss, i)
                writer.add_scalar('VAE_Loss_mah', mah_loss, i)
                # writer.add_scalar('one weight', vae.enc_fc1.weight.cpu().detach().numpy()[0][0], i)
                # writer.add_scalar('one grad', vae.enc_fc1.weight.grad.cpu().detach().numpy()[0][0], i)
                writer.add_scalar('VAE_Loss_hist', recon_loss, i)
                # writer.add_histogram('weights', vae.enc_fc1.weight.cpu().detach().numpy(), i)
                i = i + 1
                # END OF TENSOR BOARD DEBUG

            torch.save(vae.state_dict(), '%s/model_vae.pth' % save_dir)
            scheduler.step()

        print("VAE training completed...")

    vae.train(False)

    ##############
    # TRAINING MDN
    ##############

    if conf['LOAD_MDN']:
        print("loading mdn weights")
        mdn.load_state_dict(torch.load(os.path.join(save_dir, 'model_mdn.pth')))

    if conf['TRAIN_MDN']:
        print("starting MDN training...")
        mdn.train(True)
        optimizer = optim.Adam(mdn.parameters(), lr=conf['MDN_LR'])
        scheduler = StepLR(optimizer, step_size=conf['SCHED_MDN_STEP'], gamma=conf['SCHED_MDN_GAMMA'])
        i = 0
        for epochs in range(conf['EPOCHS']):
            for batch_idx, (input_color, _, _, _, grey_cropped) in tqdm(enumerate(data_loader), total=nbatches):

                # moving to cuda
                input_color = input_color.cuda()
                grey_cropped = grey_cropped.cuda()

                # network forwards
                optimizer.zero_grad()
                mu, logvar, _ = vae(color=input_color, z_in=None)
                mdn_gmm_params = mdn(grey_cropped)

                # computing mdn loss
                loss, loss_l2 = utilities.mdn_loss(
                    mdn_gmm_params,
                    mu,
                    torch.sqrt(torch.exp(logvar)),
                    conf['BATCHSIZE']
                )
                loss.backward()
                optimizer.step()

                # tensor board debug
                writer.add_scalar('MDN_Loss', loss.item(), i)
                i = i + 1

            torch.save(mdn.state_dict(), '%s/model_mdn.pth' % save_dir)
            scheduler.step()
        print("MDN training completed.")

    data, data_loader = utilities.load_data('test', outdir=os.path.join(save_dir, 'results_mdn'))
    nbatches = np.int_(np.floor(data.img_num / conf['BATCHSIZE']))

    ###################
    # TESTING VAE + MDN
    ###################

    if conf['TEST_MDN_VAE']:
        print("testing mdn + vae..")
        mdn.train(False)
        vae.eval()
        mdn.eval()
        for batch_idx, (batch, batch_recon_const, batch_weights,
                        batch_recon_const_outres, grey_cropped) in \
                tqdm(enumerate(data_loader), total=nbatches):
            with torch.no_grad():
                grey_cropped = grey_cropped.cuda()  # grey features

                mdn_gmm_params = mdn(grey_cropped)
                gmm_mu, gmm_pi = utilities.get_gmm_coeffs(mdn_gmm_params)

                # creating indexes and ordering means according to the gaussian weights Pi
                pi_indexes = gmm_pi.argsort(dim=1, descending=True)
                gmm_mu = gmm_mu.reshape(conf['BATCHSIZE'], conf['NMIX'], conf['HIDDENSIZE'])
                for i in range(conf['BATCHSIZE']):
                    gmm_mu[i, ...] = gmm_mu[i, pi_indexes[i], ...]

                # calculating results foreach sample
                result = []
                for mix in range(conf['NMIX']):
                    _, _, color_out = vae(color=None, z_in=gmm_mu[:, mix, :])
                    result.append(color_out.unsqueeze(dim=1))
                result = torch.cat(result, dim=1)  # concat over the num-mix dimension

                data.dump_results(
                    color=result,  # batch of 8 predictions  for AB channels
                    grey=batch_recon_const,  # batch of original grey channel
                    gt=batch,  # batch of gt AB channels
                    name=batch_idx,
                    nmix=conf['NMIX']
                )



    ###############
    #### CVAE  ####
    ###############

    print("creating model B architecture...")
    cvae = CVAE(conf)
    cvae.cuda()

    data, data_loader = utilities.load_data('train', outdir=os.path.join(save_dir, 'results_cvae'))
    nbatches = np.int_(np.floor(data.img_num / conf['BATCHSIZE']))

    ###############
    # TRAINING CVAE
    ###############

    if conf['LOAD_CVAE']:
        print("loading weights for CVAE")
        cvae.load_state_dict(torch.load(os.path.join(save_dir, 'model_cvae.pth')))

    if conf['TRAIN_CVAE']:
        print("starting CVAE Training for model B")
        cvae.train(True)
        optimizer = optim.Adam(cvae.parameters(), lr=conf['VAE_LR'])

        for epochs in range(conf['EPOCHS']):
            for batch_idx, (batch, batch_recon_const, batch_weights, batch_recon_const_outres, _) in \
                    tqdm(enumerate(data_loader), total=nbatches):

                input_color = batch.cuda()
                lossweights = batch_weights.cuda()
                lossweights = lossweights.view(conf['BATCHSIZE'], -1)
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
            torch.save(cvae.state_dict(), '%s/model_cvae.pth' % save_dir)

        print("CVAE training completed.")

    ##############
    # TESTING CVAE
    ##############

    if conf['TEST_CVAE']:
        print("starting CVAE testing..")
        data, data_loader = utilities.load_data('test', outdir=os.path.join(save_dir, 'results_cvae'))
        nbatches = np.int_(np.floor(data.img_num / conf['BATCHSIZE']))

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
        print("CVAE testing completed")