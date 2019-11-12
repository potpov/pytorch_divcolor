from conf import default_conf
from torch.utils.data import DataLoader
from datasets.bigearth.bigearth_dataset import BigEarthDataset
from datasets.lfw.lfw_dataset import Colordata
from torch.utils.data import SubsetRandomSampler
import os
import datetime
import json
import torch
import cv2
import numpy as np


class Utilities:

    def __init__(self, load_dir=None, dataset_name=default_conf['DATASET_NAME']):
        """
        loading config file from json file or
        creating a new experiment from the current default configuration
        default configuration is stored in conf.py
        :param load_dir: name of the folder in the experiment dir
        :return: new experiment dir or loaded experiment dir
        """
        if not load_dir:  # NEW EXPERIMENT
            # generating unique name for the experiment folder
            datalog = str(datetime.datetime.now()).replace(' ', '_')
            save_dir = os.path.join(default_conf['OUT_DIR'], dataset_name, datalog)

            # creating folders for model, config and results
            os.mkdir(save_dir)
            if default_conf['TEST_MDN_VAE']:
                os.mkdir(os.path.join(save_dir, 'results_mdn'))
            if default_conf['TEST_CVAE']:
                os.mkdir(os.path.join(save_dir, 'results_cvae'))

            # dump configuration file
            with open(os.path.join(save_dir, 'config.json'), "w") as write_file:
                json.dump(default_conf, write_file, indent=4)

            # saving class attributes
            self.save_dir = save_dir
            self.conf = default_conf
            self.dataset_name = dataset_name

        else:  # LOADING PREVIOUS EXPERIMENT
            save_dir = os.path.join(default_conf['OUT_DIR'], dataset_name, load_dir)

            # loading config file from json
            with open(os.path.join(save_dir, 'config.json'), 'r') as handle:
                config = json.load(handle)

            # create results folder if does not exists
            if config['TEST_MDN_VAE'] and not os.path.isdir(os.path.join(save_dir, 'results_mdn')):
                os.mkdir(os.path.join(save_dir, 'results_mdn'))
            if config['TEST_CVAE'] and not os.path.isdir(os.path.join(save_dir, 'results_cvae')):
                os.mkdir(os.path.join(save_dir, 'results_cvae'))

            # saving class attributes
            self.save_dir = save_dir
            self.conf = config
            self.dataset_name = dataset_name

    def load_data(self, split):
        """
        generate dataloader according to the dataset
        :param split: {train|test}
        :return: pytorch dataloader object
        """
        # BIG EARTH DATA LOADER
        if self.dataset_name == 'bigearth':
            big_earth = BigEarthDataset('big_earth_3000.csv', 'quantiles_3000.json', 42)
            train_idx, test_idx = big_earth.split_dataset(0.2)
            if split == 'train':
                sampler = SubsetRandomSampler(train_idx)
            else:
                sampler = SubsetRandomSampler(test_idx)

            data_loader = torch.utils.data.DataLoader(
                    big_earth,
                    batch_size=self.conf['BATCHSIZE'],
                    sampler=sampler,
                    num_workers=self.conf['NTHREADS'],
                    drop_last=True
                )
            return data_loader
        # LFW DATA LOADER
        elif self.dataset_name == 'lfw':
            data = Colordata(
                self.conf,
                split=split
            )

            data_loader = DataLoader(
                dataset=data,
                num_workers=self.conf['NTHREADS'],
                batch_size=self.conf['BATCHSIZE'],
                shuffle=True,
                drop_last=True
            )
            return data_loader
        # ERROR
        else:
            raise Exception('dataset not valid')

    def restore(self, img_enc):
        """
        perform conversion to RGB
        :param img_enc: CIELAB channels
        :return: RGB conversion
        """
        # img_dec = (((img_enc + 1.) * 1.) / 2.) * 255.
        img_dec = img_enc
        img_dec[img_dec < 0.] = 0.
        img_dec[img_dec > 255.] = 255.
        return img_dec.type(torch.uint8)

    def dump_results(self, color, grey, gt, nmix, model_name, file_name='result'):
        """
        :param color: network output 32x(8)x2x64x64
        :param grey: grey input 32x64x64
        :param gt: original image  32x2x64x64
        :param nmix: number of samples from the mdn
        :param name: output name for this file
        """

        # here we print the output image for the entire batch (in pieces)
        net_result = np.zeros((self.conf['BATCHSIZE'] * self.conf['IMG_H'], nmix * self.conf['IMG_W'], 3), dtype='uint8')
        border_img = 255 * np.ones((self.conf['BATCHSIZE'] * self.conf['IMG_H'], 128, 3), dtype='uint8')  # border

        # restoring previous shapes and formats
        # color = (F.interpolate(color, size=(2, self.conf['IMG_H'], self.conf['IMG_W'])))
        # grey = (F.interpolate(grey, size=(self.conf['IMG_H'], self.conf['IMG_W'])))
        # gt = (F.interpolate(gt, size=(self.conf['IMG_H'], self.conf['IMG_W'])))

        # swap axes and reshape layers to fit output image
        grey = grey.reshape((self.conf['BATCHSIZE'] * self.conf['IMG_H'], self.conf['IMG_W']))

        if nmix != 1:  # CVAE case where we haven't multiple samplings
            color = color.permute((0, 3, 1, 4, 2))
        else:
            color = color.permute((0, 2, 3, 1))

        color = color.reshape((self.conf['BATCHSIZE'] * self.conf['IMG_H'], nmix * self.conf['IMG_W'], 2))

        gt = gt.permute((0, 2, 3, 1))
        gt = gt.reshape((self.conf['BATCHSIZE'] * self.conf['IMG_H'], self.conf['IMG_W'], 2))

        gt_print = cv2.merge((self.restore(grey).data.numpy(), self.restore(gt).data.numpy()))
        net_result[:, :, 0] = self.restore(grey.repeat((1, nmix)))
        net_result[:, :, 1:3] = self.restore(color).cpu()
        net_result = cv2.cvtColor(net_result, cv2.COLOR_LAB2RGB)
        gt_print = cv2.cvtColor(gt_print, cv2.COLOR_LAB2RGB)

        out_fn_pred = os.path.join(self.save_dir, model_name, str(file_name)+'.jpg')
        cv2.imwrite(out_fn_pred, np.concatenate((net_result, border_img, gt_print), axis=1))
