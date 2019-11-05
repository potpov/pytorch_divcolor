import cv2
import numpy as np
import conf
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class colordata(Dataset):

    def __init__(self, out_directory, listdir=None, featslistdir=None, shape=(64, 64),
                 subdir=False, ext='JPEG', outshape=(256, 256), split='train'):

        self.img_fns = []
        self.feats_fns = []

        with open('%s/list.%s.vae.txt' % (listdir, split), 'r') as ftr:
            for img_fn in ftr:
                self.img_fns.append(img_fn.strip('\n'))

        with open('%s/list.%s.txt' % (featslistdir, split), 'r') as ftr:
            for feats_fn in ftr:
                self.feats_fns.append(feats_fn.strip('\n'))

        self.img_num = min(len(self.img_fns), len(self.feats_fns))
        self.shape = shape
        self.outshape = outshape
        self.out_directory = out_directory

        self.lossweights = None
        countbins = 1. / np.load('data/zhang_weights/prior_probs.npy')
        binedges = np.load('data/zhang_weights/ab_quantize.npy').reshape(2, 313)
        lossweights = {}
        for i in range(313):
            if binedges[0, i] not in lossweights:
                lossweights[binedges[0, i]] = {}
            lossweights[binedges[0, i]][binedges[1, i]] = countbins[i]
        self.binedges = binedges
        self.lossweights = lossweights

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        color_ab = np.zeros((2, self.shape[0], self.shape[1]), dtype='f')
        weights = np.ones((2, self.shape[0], self.shape[1]), dtype='f')
        recon_const = np.zeros((1, self.shape[0], self.shape[1]), dtype='f')
        recon_const_outres = np.zeros((1, self.outshape[0], self.outshape[1]), dtype='f')
        greyfeats = np.zeros((512, 28, 28), dtype='f')

        img_large = cv2.imread(self.img_fns[idx])
        if self.shape is not None:
            img = cv2.resize(img_large, (self.shape[0], self.shape[1]))
            img_outres = cv2.resize(img_large, (self.outshape[0], self.outshape[1]))

        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_lab_outres = cv2.cvtColor(img_outres, cv2.COLOR_BGR2LAB)

        img_lab = ((img_lab * 2.) / 255.) - 1.
        img_lab_outres = ((img_lab_outres * 2.) / 255.) - 1.

        recon_const[0, :, :] = img_lab[..., 0]
        recon_const_outres[0, :, :] = img_lab_outres[..., 0]

        color_ab[0, :, :] = img_lab[..., 1].reshape(1, self.shape[0], self.shape[1])
        color_ab[1, :, :] = img_lab[..., 2].reshape(1, self.shape[0], self.shape[1])

        if self.lossweights is not None:
            weights = self.__getweights__(color_ab)

        featobj = np.load(self.feats_fns[idx])
        greyfeats[:, :, :] = featobj['arr_0']

        return color_ab, recon_const, weights, recon_const_outres, greyfeats

    def __getweights__(self, img):
        img_vec = img.reshape(-1)
        img_vec = img_vec * 128.
        img_lossweights = np.zeros(img.shape, dtype='f')
        img_vec_a = img_vec[:np.prod(self.shape)]
        binedges_a = self.binedges[0, ...].reshape(-1)
        binid_a = [binedges_a.flat[np.abs(binedges_a - v).argmin()] for v in img_vec_a]
        img_vec_b = img_vec[np.prod(self.shape):]
        binedges_b = self.binedges[1, ...].reshape(-1)
        binid_b = [binedges_b.flat[np.abs(binedges_b - v).argmin()] for v in img_vec_b]
        binweights = np.array([self.lossweights[v1][v2] for v1, v2 in zip(binid_a, binid_b)])
        img_lossweights[0, :, :] = binweights.reshape(self.shape[0], self.shape[1])
        img_lossweights[1, :, :] = binweights.reshape(self.shape[0], self.shape[1])
        return img_lossweights

    def restore(self, img_enc):
        """
        perform conversion to RGB
        :param img_enc: CIELAB channels
        :return: RGB conversion
        """
        img_dec = (((img_enc + 1.) * 1.) / 2.) * 255.
        img_dec[img_dec < 0.] = 0.
        img_dec[img_dec > 255.] = 255.
        return img_dec.type(torch.uint8)

    def dump_results(self, color, grey, gt, nmix=conf.NMIX, name='result'):
        """
        :param color: network output 32x(8)x2x64x64
        :param grey: grey input 32x64x64
        :param gt: original image  32x2x64x64
        :param nmix: number of samples from the mdn
        :param name: output name for this file
        """

        # here we print the output image for the entire batch (in pieces)
        net_result = np.zeros((conf.BATCHSIZE * conf.IMG_H, nmix * conf.IMG_W, 3), dtype='uint8')
        border_img = 255 * np.ones((conf.BATCHSIZE * conf.IMG_H, 128, 3), dtype='uint8')  # border

        # restoring previous shapes and formats
        # color = (F.interpolate(color, size=(2, conf.IMG_H, conf.IMG_W)))
        # grey = (F.interpolate(grey, size=(conf.IMG_H, conf.IMG_W)))
        # gt = (F.interpolate(gt, size=(conf.IMG_H, conf.IMG_W)))

        # swap axes and reshape layers to fit output image
        grey = grey.reshape((conf.BATCHSIZE * conf.IMG_H, conf.IMG_W))

        if nmix != 1:  # CVAE case where we haven't multiple samplings
            color = color.permute((0, 3, 1, 4, 2))
        else:
            color = color.permute((0, 2, 3, 1))

        color = color.reshape((conf.BATCHSIZE * conf.IMG_H, nmix * conf.IMG_W, 2))

        gt = gt.permute((0, 2, 3, 1))
        gt = gt.reshape((conf.BATCHSIZE * conf.IMG_H, conf.IMG_W, 2))

        gt_print = cv2.merge((self.restore(grey).data.numpy(), self.restore(gt).data.numpy()))
        net_result[:, :, 0] = self.restore(grey.repeat((1, nmix)))
        net_result[:, :, 1:3] = self.restore(color).cpu()
        net_result = cv2.cvtColor(net_result, cv2.COLOR_LAB2BGR)
        gt_print = cv2.cvtColor(gt_print, cv2.COLOR_LAB2BGR)

        out_fn_pred = '%s/%s.png' % (self.out_directory, name)
        cv2.imwrite(out_fn_pred, np.concatenate((net_result, border_img, gt_print), axis=1))
        return
