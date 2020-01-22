from conf import default_conf
from torch.utils.data import DataLoader
from datasets.bigearth.bigearth_dataset import BigEarthDataset
from datasets.bigearth.quantizer import quantization
from torch.utils.data import SubsetRandomSampler
import os
import json
import torch
import cv2
import numpy as np
import shutil
import glob
import shutil


class Utilities:

    def __init__(self, load_dir, dataset_name=default_conf['DATASET_NAME'], overwrite=False):
        """
        loading config file from json file or
        creating a new experiment from the current default configuration
        default configuration is stored in conf.py
        :param load_dir: name of the folder in the experiment dir
        :param dataset_name: {bigearth|lfw}
        :return: new experiment dir or loaded experiment dir
        """

        save_dir = os.path.join(default_conf['OUT_DIR'], dataset_name, load_dir)

        # if overwrite flag is on, whatever there's on the savedir path
        # gonna be overwritten
        if os.path.exists(save_dir) and overwrite:
            print("overwriting ", save_dir)
            shutil.rmtree(save_dir)

        if not os.path.exists(save_dir):  # NEW EXPERIMENT
            print("creating new experiment with name: ", load_dir)
            # creating folders for model, config and results
            os.mkdir(save_dir)

            if default_conf['TEST_CVAE']:
                os.mkdir(os.path.join(save_dir, 'results_cvae'))

            # dump DEFAULT configuration file
            with open(os.path.join(save_dir, 'config.json'), "w") as write_file:
                json.dump(default_conf, write_file, indent=4)
            self.conf = default_conf

        else:
            print("loading previous experiment: ", load_dir)
            # loading config file from json
            with open(os.path.join(save_dir, 'config.json'), 'r') as handle:
                config = json.load(handle)

            # cleaning previous results folders
            for prev_exp in glob.glob(os.path.join(save_dir, 'results*')):
                shutil.rmtree(prev_exp)

            # create new results folders
            if config['TEST_MDN_VAE'] and not os.path.isdir(os.path.join(save_dir, 'results_mdn')):
                os.mkdir(os.path.join(save_dir, 'results_mdn'))
            if config['TEST_CVAE'] and not os.path.isdir(os.path.join(save_dir, 'results_cvae')):
                os.mkdir(os.path.join(save_dir, 'results_cvae'))

            self.conf = config

        # saving class attributes
        self.title = load_dir
        self.save_dir = save_dir
        self.dataset_name = dataset_name

    def epoch_checkpoint(self, model, epoch):
        """
        update the json file with the current achieved epoch number and print it to the config file
        epoch: epoch number
        """
        if model == 'MDN':
            self.conf['LOAD_MDN'] = True
            self.conf['MDN_EPOCH_CHECKPOINT'] = epoch
        elif model == 'VAE':
            self.conf['LOAD_VAE'] = True
            self.conf['VAE_EPOCH_CHECKPOINT'] = epoch
        elif model == 'CVAE':
            self.conf['LOAD_CVAE'] = True
            self.conf['CVAE_EPOCH_CHECKPOINT'] = epoch
        else:
            raise Exception('invalid model in epoch checkpoint!')
        # saving the new configuration
        with open(os.path.join(self.save_dir, 'config.json'), "w") as write_file:
            json.dump(self.conf, write_file, indent=4)

    def test_complete(self):
        self.conf['LOAD_CVAE'] = False
        self.conf['LOAD_MDN'] = False
        self.conf['LOAD_VAE'] = False
        with open(os.path.join(self.save_dir, 'config.json'), "w") as write_file:
            json.dump(self.conf, write_file, indent=4)

    def load_data(self, split, mode, rgb, writer=None):
        """
        generate dataloader according to the dataset
        :param split: {train|test}
        :return: pytorch dataloader object
        """
        # BIG EARTH DATA LOADER
        if self.dataset_name == 'bigearth':

            big_earth = BigEarthDataset(
                self.conf['BIG_EARTH_CVS_NAME'],
                42,
                self.conf['BANDS'],
                mode=mode,
                RGB=rgb,
                weights_file=self.conf['WEIGHT_FILENAME']
            )
            train_idx, val_idx, test_idx = big_earth.split_dataset(0.2, 0.3)

            if split == 'train':
                sampler = SubsetRandomSampler(train_idx)
                batchsize = self.conf['BATCHSIZE']
            else:
                sampler = SubsetRandomSampler(test_idx)
                batchsize = self.conf['TEST_BATCHSIZE']

            data_loader = torch.utils.data.DataLoader(
                big_earth,
                batch_size=batchsize,
                sampler=sampler,
                num_workers=self.conf['NTHREADS'],
                drop_last=True,
            )
            return data_loader
        else:
            raise Exception('dataset not valid')

    def reload_weights(self):
        # calculate histogram for this dataset
        quantization(conf=self.conf, q_factor=self.conf['Q_FACTOR'])

    def restore(self, img_enc):
        """
        perform conversion to RGB
        :param img_enc: CIELAB channels
        :return: RGB conversion
        """
        img_dec = (((img_enc + 1.) * 1.) / 2.) * 255.
        # img_dec = img_enc
        img_dec[img_dec < 0.] = 0.
        img_dec[img_dec > 255.] = 255.
        return img_dec.type(torch.uint8)

    def generate_header(self, bordershape, with_posterior=True):
        font = cv2.FONT_HERSHEY_SIMPLEX

        column = 2 + self.conf['NMIX'] if with_posterior else 1 + self.conf['NMIX']
        borders = 2 * bordershape if with_posterior else bordershape
        header = 255 * np.ones((40, self.conf['IMG_W'] * column + borders, 3))

        # print gt same for both cases
        cv2.putText(header, 'GT', (10, 25), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        if with_posterior:
            cv2.putText(header, 'posterior', (self.conf['IMG_W'] + bordershape, 25), font, 0.5, (0, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(header, 'samples',
                        ((2 + int(self.conf['NMIX'] / 2)) * self.conf['IMG_W'] + 2 * bordershape, 25),
                        font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(header, 'samples', (self.conf['IMG_W'] + bordershape, 25), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        return header

    def dump_results(self, color, grey, gt, nmix, model_name, file_name='result', tb_writer=None, posterior=None):
        """
        :param color: network output 32x(8)x2x64x64
        :param grey: grey input 32x64x64
        :param gt: original image  32x2x64x64
        :param nmix: number of samples from the mdn
        :param model_name: {mdn|cvae}
        :param file_name: name of the new image
        :param tb_writer: tensorboardX writer
        :param posterior: output image if decoder sample from posterior
        """

        # here we print the output image for the entire batch (in pieces)
        net_result = np.zeros((self.conf['TEST_BATCHSIZE'] * self.conf['IMG_H'], nmix * self.conf['IMG_W'], 3),
                              dtype='uint8')
        # black stripe between sections
        border_img = 0 * np.ones((self.conf['TEST_BATCHSIZE'] * self.conf['IMG_H'], 10, 3))

        # swap axes and reshape layers to fit correct format when reshaping
        grey = grey.reshape((self.conf['TEST_BATCHSIZE'] * self.conf['IMG_H'], self.conf['IMG_W']))

        if nmix != 1:
            color = color.permute((0, 3, 1, 4, 2))
        else:
            color = color.squeeze(1)
            color = color.permute((0, 2, 3, 1))

        color = color.reshape((self.conf['TEST_BATCHSIZE'] * self.conf['IMG_H'], nmix * self.conf['IMG_W'], 2))
        gt = gt.permute((0, 2, 3, 1))
        gt = gt.reshape((self.conf['TEST_BATCHSIZE'] * self.conf['IMG_H'], self.conf['IMG_W'], 2))

        # LAB -> RGB
        gt_print = cv2.merge((self.restore(grey).data.numpy(), self.restore(gt).data.numpy()))
        net_result[:, :, 0] = self.restore(grey.repeat((1, nmix)))
        net_result[:, :, 1:3] = self.restore(color).detach().cpu()
        gt_print = cv2.cvtColor(gt_print, cv2.COLOR_LAB2BGR)
        net_result = cv2.cvtColor(net_result, cv2.COLOR_LAB2BGR)

        if posterior is not None:
            # one more colomn for posterior
            posterior = posterior.permute((0, 2, 3, 1))
            posterior = posterior.reshape((self.conf['TEST_BATCHSIZE'] * self.conf['IMG_H'], self.conf['IMG_W'], 2))
            posterior = cv2.merge((self.restore(grey).data.numpy(), self.restore(posterior).cpu().data.numpy()))
            posterior = cv2.cvtColor(posterior, cv2.COLOR_LAB2BGR)
            result_image = np.concatenate((gt_print, border_img, posterior, border_img, net_result), axis=1)
        else:
            result_image = np.concatenate((gt_print, border_img, net_result), axis=1)

        header = self.generate_header(border_img.shape[1], with_posterior=(posterior is not None))
        result_image = np.concatenate((header, result_image), axis=0)

        # create output dir if not exists and save result on disk
        out_fn_pred = os.path.join(self.save_dir, model_name, str(file_name) + '.jpg')
        if not os.path.exists(os.path.join(self.save_dir, model_name)):
            os.mkdir(os.path.join(self.save_dir, model_name))
        cv2.imwrite(out_fn_pred, result_image)

        # saving path for tensorboard will be something like mdn/result on iteration == batch idx
        # tensorboard needs input as 3 x H x W
        if tb_writer is not None:
            result_image = result_image[:, :, ::-1]
            result_image = np.transpose(result_image.astype('uint8'), (2, 0, 1))
            tb_writer.add_image('{}/img_results'.format(model_name), result_image, int(file_name))
