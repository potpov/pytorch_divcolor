from networks.cvae import CVAE
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from loss import Losses
import os
import numpy as np


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().max())
    return ave_grads, layers


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

    def train(self, data_loader, test_set, writer):
        print("starting CVAE Training..")
        self.cvae.train(True)
        optimizer = optim.SGD(self.cvae.parameters(), lr=self.conf['CVAE_LR'])
        # warm up conf
        warm_up = np.ones(shape=self.conf['EPOCHS'])
        warm_up[0:int(self.conf['EPOCHS'] * 0.5)] = np.linspace(self.conf['WARM_UP_TH'], 1, num=(self.conf['EPOCHS'] * 0.5))

        for epochs in range(self.checkpoint, self.conf['EPOCHS']):
            for idx, (spectrals, (input_color, grey_little, batch_weights, _)) in \
                    tqdm(enumerate(data_loader), total=len(data_loader)):
                input_color = input_color.cuda()
                input_color.requires_grad_()
                lossweights = batch_weights.cuda()
                lossweights = lossweights.view(self.conf['BATCHSIZE'], -1)
                input_spectral = spectrals.permute(0, 3, 1, 2).float().cuda()

                optimizer.zero_grad()
                color_out, mu, logvar = self.cvae(color=input_color, inputs=input_spectral)
                # fancy LOSS Calculation, weighted on 'CVAE_LOSS'
                hist_loss, kl_loss, grad_penalty = self.loss_set.cvae_loss(
                    color_out,
                    input_color,
                    lossweights,
                    mu,
                    logvar
                )

                loss = hist_loss + (kl_loss + grad_penalty) * warm_up[epochs]

                # log loss
                writer.add_scalar('CVAE/prior', kl_loss, epochs*len(data_loader) + idx)
                writer.add_scalar('CVAE/hist', hist_loss.item(), epochs*len(data_loader) + idx)
                writer.add_scalar('CVAE/grad_penalty', grad_penalty.item(), epochs*len(data_loader) + idx)
                writer.add_scalar('CVAE/final', loss.item(), epochs*len(data_loader) + idx)

                loss.backward()
                # nn.utils.clip_grad_value_(self.cvae.parameters(), self.conf['CLIP_TH'])
                optimizer.step()

            # validation test
            if epochs % self.conf['TEST_ON_TRAIN_RATE'] == 0 and epochs != 0:
                print("\nexecuting validation on train epoch: " + str(epochs))
                self.test(test_set, writer, output_name='results_epoch{}'.format(str(epochs)))
                print("\nvalidation completed, back to training")
                self.cvae.train(True)

            # saving weights and checkpoints
            self.utilities.epoch_checkpoint('CVAE', epochs)
            torch.save(self.cvae.state_dict(), '%s/model_cvae.pth' % self.save_dir)

    def test(self, data_loader, writer, output_name="results_cvae"):
        print("starting CVAE testing..")
        self.cvae.train(False)
        self.cvae.eval()

        with torch.no_grad():
            for idx, (spectrals, (batch, grey_little, batch_weights, _)) in \
                    tqdm(enumerate(data_loader), total=len(data_loader)):

                input_color = batch.cuda()
                spectrals = spectrals.permute(0, 3, 1, 2).float().cuda()

                # checking result if encoder samples from posterior (gran truth)
                posterior, _, _ = self.cvae(color=input_color, inputs=spectrals)

                # checking results if encoder samples from prior (NMIX samplings from gaussian)
                results = []
                for i in range(self.conf['NMIX']):
                    color_out, _, _ = self.cvae(color=None, inputs=spectrals)
                    results.append(color_out.unsqueeze(1))
                results = torch.cat(results, dim=1)

                self.utilities.dump_results(
                    color=results,  # batch of 8 predictions  for AB channels
                    grey=grey_little,  # batch of original grey channel
                    gt=batch,  # batch of gt AB channels
                    file_name=idx,
                    nmix=self.conf['NMIX'],
                    model_name=output_name,
                    tb_writer=writer,
                    posterior=posterior
                )

        print("CVAE testing completed")
