from networks.resnet.Resnet18 import CVAE
# from networks.cvae_skips import CVAE
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from tqdm import tqdm
from loss import Losses
import os
import numpy as np
import random
from sklearn.metrics import average_precision_score
import cv2
from datasets.bigearth.bigearth_dataset import lab2rgb


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().max())
    return ave_grads, layers


class Cvae:

    def __init__(self, utilities, writer):
        # load config file from previous or new experiments
        self.utilities = utilities
        self.save_dir = utilities.save_dir
        self.conf = utilities.conf
        self.writer = writer
        # creating loss class
        self.loss_set = Losses(self.conf)
        self.cvae = CVAE(self.conf).cuda()
        self.checkpoint = 0

    def load_weights(self):
        print("loading CVAE weights, starting from epoch: " + str(self.conf['CVAE_EPOCH_CHECKPOINT']))
        self.checkpoint = self.conf['CVAE_EPOCH_CHECKPOINT']
        self.cvae.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_cvae.pth')), strict=False)

    def train(self, data_loader, val_loader):
        print("starting CVAE Training..")
        set_random_seed(128)
        self.cvae.train(True)
        optimizer = optim.SGD(self.cvae.parameters(), lr=self.conf['CVAE_LR'])
        # warm up conf
        warm_up = np.ones(shape=self.conf['EPOCHS'] * len(data_loader))
        warm_up[0:int(self.conf['EPOCHS'] * 0.7 * len(data_loader))] = np.linspace(
            self.conf['WARM_UP_TH'],
            1,
            num=(self.conf['EPOCHS'] * 0.7 * len(data_loader))
        )

        for epochs in range(self.checkpoint, self.conf['EPOCHS']):
            for idx, (spectrals, (input_color, grey_little, batch_weights)) in \
                    tqdm(enumerate(data_loader), total=len(data_loader)):

                spectrals = spectrals.cuda()
                input_color = input_color.cuda()
                # input_color.requires_grad_()
                lossweights = batch_weights.cuda()
                lossweights = lossweights.view(self.conf['BATCHSIZE'], -1)

                optimizer.zero_grad()
                color_out, mu, logvar, z_grey, z_color, z = self.cvae(color=input_color, inputs=spectrals)

                hist_loss, kl_loss = self.loss_set.cvae_loss(
                    color_out,
                    input_color,
                    lossweights,
                    mu,
                    logvar
                )

                loss = hist_loss + kl_loss * warm_up[epochs*len(data_loader)+idx]
                if torch.isnan(loss):
                    print("\n\nabort: nan on epoch: ", epochs, idx)
                    print("val hist loss: ", hist_loss)
                    print("val kl: ", kl_loss)
                    exit()

                # log loss
                self.writer.add_scalar('CVAE/prior', kl_loss, epochs*len(data_loader) + idx)
                self.writer.add_scalar('CVAE/hist', hist_loss.item(), epochs*len(data_loader) + idx)
                self.writer.add_scalar('CVAE/final', loss.item(), epochs*len(data_loader) + idx)
                self.writer.add_histogram('z_hist', z[0].cpu().detach().numpy(), epochs*len(data_loader) + idx)
                self.writer.add_histogram('z_grey_hist', z_grey[0].cpu().detach().numpy(), epochs*len(data_loader) + idx)
                self.writer.add_histogram('z_color_hist', z_color[0].cpu().detach().numpy(), epochs*len(data_loader) + idx)
                loss.backward()
                # nn.utils.clip_grad_value_(self.cvae.parameters(), self.conf['CLIP_TH'])
                optimizer.step()

            # validation test
            if epochs % self.conf['TEST_ON_TRAIN_RATE'] == 0 and epochs != 0:
                print("\nexecuting validation on train epoch: " + str(epochs))
                self.test(val_loader, output_name='results_epoch{}'.format(str(epochs)))
                print("\nvalidation completed, back to training")
                self.cvae.train(True)

            # saving weights and checkpoints
            self.utilities.epoch_checkpoint('CVAE', epochs)
            torch.save(self.cvae.state_dict(), '%s/model_cvae.pth' % self.save_dir)

    def test(self, data_loader, output_name="results_cvae"):
        print("starting CVAE testing..")
        self.cvae.train(False)
        self.cvae.eval()

        with torch.no_grad():
            for idx, (spectrals, (batch, grey_little, batch_weights)) in \
                    tqdm(enumerate(data_loader), total=len(data_loader)):

                input_color = batch.cuda()
                spectrals = spectrals.cuda()

                # checking result if encoder samples from posterior (gran truth)
                posterior, _, _, _, _, _ = self.cvae(color=input_color, inputs=spectrals)

                # n = min(posterior.size(0), 16)
                # recon_rgb_n = lab2rgb(grey_little[:n], posterior[:n])
                # rgb_n = lab2rgb(grey_little[:n], input_color[:n])
                # comparison = torch.cat([rgb_n[:n], recon_rgb_n[:n]])
                # writer.add_images(output_name, comparison, idx)
                # checking results if encoder samples from prior (NMIX samplings from gaussian)

                results = []
                for i in range(self.conf['NMIX']):
                    color_out, _, _, _, _, _ = self.cvae(color=None, inputs=spectrals)
                    results.append(color_out.unsqueeze(1))
                results = torch.cat(results, dim=1)

                self.utilities.dump_results(
                    color=results,  # batch of 8 predictions  for AB channels
                    grey=grey_little,  # batch of original grey channel
                    gt=batch,  # batch of gt AB channels
                    file_name=idx,
                    nmix=self.conf['NMIX'],
                    model_name=output_name,
                    tb_writer=self.writer,
                    posterior=posterior
                )
        print("CVAE testing completed")