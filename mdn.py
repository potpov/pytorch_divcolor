from networks.vae import VAE
from networks.mdn import MDN
import torch
import torch.optim as optim
from tqdm import tqdm
from loss import Losses
import os
from torch.optim.lr_scheduler import StepLR


class Mdn:

    def __init__(self, utilities):
        # load config file from previous or new experiments
        self.utilities = utilities
        self.save_dir = utilities.save_dir
        self.conf = utilities.conf
        # creating loss class
        self.loss_set = Losses(self.conf)
        self.vae = VAE(self.conf).cuda()
        self.mdn = MDN(self.conf).cuda()
        self.checkpoint = 0

    def load_vae_weights(self):
        print("loading vae weights. starting from epoch: " + str(self.conf['VAE_EPOCH_CHECKPOINT']))
        self.checkpoint = self.conf['VAE_EPOCH_CHECKPOINT']
        self.vae.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_vae.pth')))

    def train_vae(self, data_loader, writer):
        conf = self.conf
        print("starting VAE Training for model A")
        self.vae.train(True)
        optimizer = optim.Adam(self.vae.parameters(), lr=conf['VAE_LR'])
        scheduler = StepLR(optimizer, step_size=conf['SCHED_VAE_STEP'], gamma=conf['SCHED_VAE_GAMMA'])
        i = 0

        for epochs in range(self.checkpoint, conf['EPOCHS']):
            for batch_idx, (input_color, _, weights, _, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
                # moving to cuda
                input_color = input_color.cuda()
                lossweights = weights.cuda()
                lossweights = lossweights.view(conf['BATCHSIZE'], -1)

                # model forward
                optimizer.zero_grad()
                mu, logvar, color_out = self.vae(color=input_color, z_in=None)

                # computing losses
                kl_loss, hist_loss, grad_loss, mah_loss = self.loss_set.vae_loss(
                    mu,
                    logvar,
                    color_out,
                    input_color,
                    lossweights
                )
                # summing losses
                loss = sum([
                    kl_loss.mul(conf['KL_W']),
                    hist_loss.mul(conf['HIST_W']),
                    grad_loss.mul(conf['GRA_W']),
                    mah_loss.mul(conf['MAH_W'])
                ]
                )

                loss.backward()
                optimizer.step()

                # TENSOR BOARD DEBUG
                writer.add_scalar('VAE/total_Loss', loss.item(), i)
                writer.add_scalar('VAE/Loss_grad', grad_loss, i)
                writer.add_scalar('VAE/Loss_kl', kl_loss, i)
                writer.add_scalar('VAE/Loss_mah', mah_loss, i)
                writer.add_scalar('VAE/Loss_hist', hist_loss, i)
                i = i + 1
                # END OF TENSOR BOARD DEBUG

            self.utilities.epoch_checkpoint('VAE', epochs)
            torch.save(self.vae.state_dict(), '%s/model_vae.pth' % self.save_dir)
            scheduler.step()
        print("VAE training completed...")
        self.vae.train(False)

    def load_mdn_weights(self):
        if self.conf['LOAD_MDN']:
            print("loading mdn weights, starting from epoch: " + str(self.conf['MDN_EPOCH_CHECKPOINT']))
            self.checkpoint = self.conf['MDN_EPOCH_CHECKPOINT']
            self.mdn.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_mdn.pth')))

    def train_mdn(self, data_loader, writer):

        print("starting MDN training...")
        self.mdn.train(True)
        optimizer = optim.Adam(self.mdn.parameters(), lr=self.conf['MDN_LR'])
        scheduler = StepLR(optimizer, step_size=self.conf['SCHED_MDN_STEP'], gamma=self.conf['SCHED_MDN_GAMMA'])
        i = 0
        for epochs in range(self.checkpoint, self.conf['EPOCHS']):
            for batch_idx, (input_color, _, _, _, grey_cropped) in tqdm(enumerate(data_loader), total=len(data_loader)):

                # moving to cuda
                input_color = input_color.cuda()
                grey_cropped = grey_cropped.cuda()

                # network forwards
                optimizer.zero_grad()
                mu, logvar, _ = self.vae(color=input_color, z_in=None)
                mdn_gmm_params = self.mdn(grey_cropped)

                # computing mdn loss
                loss, loss_l2 = self.loss_set.mdn_loss(
                    mdn_gmm_params,
                    mu,
                    torch.sqrt(torch.exp(logvar)),
                    self.conf['BATCHSIZE']
                )

                loss.backward()
                optimizer.step()

                # tensor board debug
                writer.add_scalar('MDN/total_Loss', loss.item(), i)
                i = i + 1

            self.utilities.epoch_checkpoint('MDN', epochs)
            torch.save(self.mdn.state_dict(), '%s/model_mdn.pth' % self.save_dir)
            scheduler.step()
        print("MDN training completed.")

    def test(self, data_loader, writer):
        print("testing mdn + vae..")
        self.mdn.train(False)
        self.vae.eval()
        self.mdn.eval()
        for batch_idx, (input_color, grey_little, _, _, grey_cropped) in tqdm(enumerate(data_loader),
                                                                              total=len(data_loader)):
            with torch.no_grad():
                grey_cropped = grey_cropped.cuda()  # grey features

                mdn_gmm_params = self.mdn(grey_cropped)
                gmm_mu, gmm_pi = self.loss_set.get_gmm_coeffs(mdn_gmm_params)

                # creating indexes and ordering means according to the gaussian weights Pi
                pi_indexes = gmm_pi.argsort(dim=1, descending=True)
                gmm_mu = gmm_mu.reshape(self.conf['BATCHSIZE'], self.conf['NMIX'], self.conf['HIDDENSIZE'])
                for i in range(self.conf['BATCHSIZE']):
                    gmm_mu[i, ...] = gmm_mu[i, pi_indexes[i], ...]

                # calculating results foreach sample
                result = []
                for mix in range(self.conf['NMIX']):
                    _, _, color_out = self.vae(color=None, z_in=gmm_mu[:, mix, :])
                    result.append(color_out.unsqueeze(dim=1))
                result = torch.cat(result, dim=1)  # concat over the num-mix dimension

                # printing results for this batch
                self.utilities.dump_results(
                    color=result,  # batch of 8 predictions  for AB channels
                    grey=grey_little,  # batch of original grey channel
                    gt=input_color,  # batch of gt AB channels
                    file_name=batch_idx,
                    nmix=self.conf['NMIX'],
                    model_name='results_mdn',
                    tb_writer=writer
                )
