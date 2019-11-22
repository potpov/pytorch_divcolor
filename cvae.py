from networks.cvae import CVAE
import torch
import torch.optim as optim
from tqdm import tqdm
from loss import Losses
import os
import numpy as np

class Cvae:

    def __init__(self, utilities):
        # load config file from previous or new experiments
        self.utilities = utilities
        self.save_dir = utilities.save_dir
        self.conf = utilities.conf
        # creating loss class
        self.loss_set = Losses(self.conf)
        self.cvae = CVAE(self.conf).cuda()
        self.checkpoint = 0

    def load_weights(self):
        print("loading CVAE weights, starting from epoch: " + str(self.conf['CVAE_EPOCH_CHECKPOINT']))
        self.checkpoint = self.conf['CVAE_EPOCH_CHECKPOINT']
        self.cvae.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_cvae.pth')))

    def train(self, data_loader, writer):
        print("starting CVAE Training..")
        self.cvae.train(True)
        optimizer = optim.Adam(self.cvae.parameters(), lr=self.conf['CVAE_LR'])
        i = 0
        # warm up conf
        tot_range = len(data_loader) * self.conf['EPOCHS']
        warm_up = np.ones(shape=tot_range)
        warm_up[0:int(tot_range * 0.5)] = np.linspace(0, 1, num=(tot_range * 0.5))

        for epochs in range(self.checkpoint, self.conf['EPOCHS']):
            for batch_idx, (input_color, grey_little, batch_weights, _) in \
                    tqdm(enumerate(data_loader), total=len(data_loader)):
                input_color = input_color.cuda()
                lossweights = batch_weights.cuda()
                lossweights = lossweights.view(self.conf['BATCHSIZE'], -1)
                input_grey = grey_little.cuda()

                optimizer.zero_grad()
                color_out, mu, logvar = self.cvae(color=input_color, greylevel=input_grey)
                # fancy LOSS Calculation, weighted on 'CVAE_LOSS'
                hist_loss, kl_loss = self.loss_set.cvae_loss(
                    color_out,
                    input_color,
                    lossweights,
                    mu,
                    logvar
                )
                loss = hist_loss + kl_loss.mul(warm_up[i])

                # log loss
                writer.add_scalar('CVAE/prior', kl_loss.item(), i)
                writer.add_scalar('CVAE/hist', hist_loss.item(), i)
                writer.add_scalar('CVAE/final', loss.item(), i)
                i = i + 1

                loss.backward()
                optimizer.step()
            self.utilities.epoch_checkpoint('CVAE', epochs)
            torch.save(self.cvae.state_dict(), '%s/model_cvae.pth' % self.save_dir)

    def test(self, data_loader, writer):
        print("starting CVAE testing..")

        for batch_idx, (batch, grey_little, batch_weights, _) in \
                tqdm(enumerate(data_loader), total=len(data_loader)):

            input_color = batch.cuda()
            input_grey = grey_little.cuda()

            # checking result if encoder samples from posterior (gran truth)
            self.cvae.train(True)
            posterior, _, _ = self.cvae(color=input_color, greylevel=input_grey)
            self.cvae.train(False)
            self.cvae.eval()

            # checking results if encoder samples from prior (NMIX samplings from gaussian)
            results = []
            for i in range(self.conf['NMIX']):
                color_out, _, _ = self.cvae(color=None, greylevel=input_grey)
                results.append(color_out.unsqueeze(1))
            results = torch.cat(results, dim=1)

            self.utilities.dump_results(
                color=results,  # batch of 8 predictions  for AB channels
                grey=grey_little,  # batch of original grey channel
                gt=batch,  # batch of gt AB channels
                file_name=batch_idx,
                nmix=self.conf['NMIX'],
                model_name='results_cvae',
                tb_writer=writer,
                posterior=posterior
            )
        print("CVAE testing completed")
