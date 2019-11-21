from networks.cvae import CVAE
import torch
import torch.optim as optim
from tqdm import tqdm
from loss import Losses
import os

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
        for epochs in range(self.checkpoint, self.conf['EPOCHS']):
            for batch_idx, (input_color, grey_little, batch_weights, _, _) in \
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
                loss = hist_loss + kl_loss

                # log loss
                writer.add_scalar('CVAE/prior', kl_loss.item(), i)
                writer.add_scalar('CVAE/hist', hist_loss.item(), i)
                writer.add_scalar('CVAE/final', loss.item(), i)
                i = i + 1

                loss.backward()
                optimizer.step()
            self.utilities.epoch_checkpoint('CVAE', epochs)
            torch.save(self.cvae.state_dict(), '%s/model_cvae.pth' % self.save_dir)

    def check_train(self, dataloader, writer):

        print("checking if CVAE learn something..")
        color_ab, grey_little, batch_weights, _, _ = next(iter(dataloader))

        input_color = color_ab.cuda()
        input_grey = grey_little.cuda()

        color_out, _, _ = self.cvae(color=input_color, greylevel=input_grey)

        self.utilities.dump_results(
            color=color_out,  # batch of 8 predictions  for AB channels
            grey=grey_little,  # batch of original grey channel
            gt=color_ab,  # batch of gt AB channels
            file_name=9999,
            nmix=1,
            model_name='results_cvae',
            tb_writer=writer
        )
        print("check done.")

    def test(self, data_loader, writer):
        print("starting CVAE testing..")
        self.cvae.train(False)
        for batch_idx, (batch, grey_little, batch_weights, _, _) in \
                tqdm(enumerate(data_loader), total=len(data_loader)):

            # input_feats = batch_feats.cuda()
            input_grey = grey_little.cuda()

            color_out, _, _ = self.cvae(color=None, greylevel=input_grey)

            self.utilities.dump_results(
                color=color_out,  # batch of 8 predictions  for AB channels
                grey=grey_little,  # batch of original grey channel
                gt=batch,  # batch of gt AB channels
                file_name=batch_idx,
                nmix=1,
                model_name='results_cvae',
                tb_writer=writer
            )
        print("CVAE testing completed")
