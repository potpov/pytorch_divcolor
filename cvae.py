from networks.cvae_skips import CVAE
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from loss import Losses
import os
import numpy as np
import random
from sklearn.metrics import average_precision_score
import cv2


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
        self.cvae.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_cvae.pth')), strict=False)

    def train(self, data_loader, test_set, writer):
        print("starting CVAE Training..")
        set_random_seed(128)
        self.cvae.train(True)
        optimizer = optim.SGD(self.cvae.parameters(), lr=self.conf['CVAE_LR'])
        # warm up conf
        warm_up = np.ones(shape=self.conf['EPOCHS'])
        warm_up[0:int(self.conf['EPOCHS'] * 0.5)] = np.linspace(self.conf['WARM_UP_TH'], 1, num=(self.conf['EPOCHS'] * 0.5))

        for epochs in range(self.checkpoint, self.conf['EPOCHS']):
            for idx, (spectrals, (input_color, grey_little, batch_weights)) in \
                    tqdm(enumerate(data_loader), total=len(data_loader)):

                # removing RGB bands
                nocolorbands = torch.arange(3, len(self.conf['BANDS']))
                spectrals = torch.index_select(input=spectrals, dim=1, index=nocolorbands)

                # moving bounds to [0-255] rather than [-1, 1]
                spectrals_norm = np.zeros_like(spectrals.numpy())
                spectrals_norm = cv2.normalize(spectrals.float().numpy(), spectrals_norm, 0, 255, cv2.NORM_MINMAX)
                spectrals_norm = torch.Tensor(spectrals_norm).cuda()
                optimizer.zero_grad()

                input_color = input_color.cuda()
                input_color.requires_grad_()
                lossweights = batch_weights.cuda()
                lossweights = lossweights.view(self.conf['BATCHSIZE'], -1)

                optimizer.zero_grad()
                color_out, mu, logvar = self.cvae(color=input_color, inputs=spectrals_norm)
                # fancy LOSS Calculation, weighted on 'CVAE_LOSS'
                hist_loss, kl_loss, grad_penalty = self.loss_set.cvae_loss(
                    color_out,
                    input_color,
                    lossweights,
                    mu,
                    logvar
                )

                loss = hist_loss + kl_loss * warm_up[epochs] + grad_penalty

                # log loss
                writer.add_scalar('CVAE/prior', kl_loss, epochs*len(data_loader) + idx)
                writer.add_scalar('CVAE/hist', hist_loss.item(), epochs*len(data_loader) + idx)
                writer.add_scalar('CVAE/grad_penalty', grad_penalty.item(), epochs*len(data_loader) + idx)
                writer.add_scalar('CVAE/final', loss.item(), epochs*len(data_loader) + idx)
                writer.add_scalar('CVAE/warmup', warm_up[epochs], epochs*len(data_loader) + idx)

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
            for idx, (spectrals, (batch, grey_little, batch_weights)) in \
                    tqdm(enumerate(data_loader), total=len(data_loader)):

                # removing RGB bands
                nocolorbands = torch.arange(3, len(self.conf['BANDS']))
                spectrals = torch.index_select(input=spectrals, dim=1, index=nocolorbands)

                # moving bounds to [0-255] rather than [-1, 1]
                spectrals_norm = np.zeros_like(spectrals.numpy())
                spectrals_norm = cv2.normalize(spectrals.float().numpy(), spectrals_norm, 0, 255, cv2.NORM_MINMAX)
                spectrals_norm = torch.Tensor(spectrals_norm).cuda()

                input_color = batch.cuda()
                # spectrals = spectrals.float().cuda()

                # checking result if encoder samples from posterior (gran truth)
                posterior, _, _ = self.cvae(color=input_color, inputs=spectrals_norm)

                # checking results if encoder samples from prior (NMIX samplings from gaussian)
                results = []
                for i in range(self.conf['NMIX']):
                    color_out, _, _ = self.cvae(color=None, inputs=spectrals_norm)
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

    def transfer_learning_train(self, data_loader, writer, feature_extr=True):

        print("transfer learning prediction (training)")
        optimizer = optim.SGD(self.cvae.parameters(), lr=self.conf['PREDICTION_LR'])
        loss_fn = nn.MultiLabelSoftMarginLoss()

        if feature_extr:
            for param in self.cvae.parameters():
                param.requires_grad = False
            for p in self.cvae.class_fc.parameters():
                p.requires_grad = True

        for epochs in range(0, self.conf['TRANSF_LEARNING_EPOCH']):
            for idx, (spectrals, labels, _) in \
                    tqdm(enumerate(data_loader), total=len(data_loader)):

                # removing RGB bands
                nocolorbands = torch.arange(3, len(self.conf['BANDS']))
                spectrals = torch.index_select(input=spectrals, dim=1, index=nocolorbands)

                # moving bounds to [0-255] rather than [-1, 1]
                spectrals_norm = np.zeros_like(spectrals.numpy())
                spectrals_norm = cv2.normalize(spectrals.float().numpy(), spectrals_norm, 0, 255, cv2.NORM_MINMAX)
                spectrals_norm = torch.Tensor(spectrals_norm).cuda()
                optimizer.zero_grad()

                pred, pred_sigmoid = self.cvae(color=None, inputs=spectrals_norm, prediction=True)

                loss = loss_fn(pred.cpu(), labels.float())
                loss.backward()
                writer.add_scalar('TL/loss', loss, epochs*len(data_loader) + idx)

                optimizer.step()

        print("transfer learning completed. saving weights")
        torch.save(self.cvae.state_dict(), '%s/finetuning.pth' % self.save_dir)

    def transfer_learning_test(self, data_loader, writer, feature_extr=True):

        print("TEST - prediction")

        self.cvae.train(False)
        self.cvae.eval()
        avg_pr_micro = 0

        with torch.no_grad():
            for idx, (spectrals, labels, _) in \
                    tqdm(enumerate(data_loader), total=len(data_loader)):

                # removing RGB bands
                nocolorbands = torch.arange(3, len(self.conf['BANDS']))
                spectrals = torch.index_select(input=spectrals, dim=1, index=nocolorbands)

                # moving bounds to [0-255] rather than [-1, 1]
                spectrals_norm = np.zeros_like(spectrals.numpy())
                spectrals_norm = cv2.normalize(spectrals.float().numpy(), spectrals_norm, 0, 255, cv2.NORM_MINMAX)
                spectrals_norm = torch.Tensor(spectrals_norm).cuda()

                pred, pred_sigmoid = self.cvae(color=None, inputs=spectrals_norm, prediction=True)

                avg_pr_micro = average_precision_score(
                    labels,
                    pred_sigmoid.cpu().detach(),
                    average='micro'
                )
                writer.add_scalar('TL/ap_score', avg_pr_micro, idx)

        print("final avarage precision score: ", avg_pr_micro)