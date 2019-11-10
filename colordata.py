import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Colordata(Dataset):

    def __init__(self, out_directory, conf, listdir=None, shape=(64, 64), outshape=(256, 256), split='train'):

        self.img_fns = []
        self.conf = conf
        
        with open('%s/list.%s.vae.txt' % (listdir, split), 'r') as ftr:
            for img_fn in ftr:
                self.img_fns.append(img_fn.strip('\n'))

        self.img_num = len(self.img_fns)
        self.shape = shape
        self.outshape = outshape
        self.out_directory = out_directory

        # histogram weights
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
        grey_little = np.zeros((1, self.shape[0], self.shape[1]), dtype='f')
        grey_big = np.zeros((1, self.outshape[0], self.outshape[1]), dtype='f')
        grey_cropped = np.zeros((1, 224, 224), dtype='f')

        # converting original image to CIELAB
        img = cv2.imread(self.img_fns[idx])
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_lab = ((img_lab * 2.) / 255.) - 1.

        # creating scaled versions of the image
        img_big = cv2.resize(img_lab, (self.outshape[0], self.outshape[0]))
        img_little = cv2.resize(img_lab, (self.conf['SCALED_W'], self.conf['SCALED_H']))
        img_cropped = cv2.resize(img_lab, (224, 224))

        # copying grey scale layers
        grey_cropped[0, :, :] = img_cropped[..., 0]
        grey_little[0, :, :] = img_little[..., 0]
        grey_big[0, :, :] = img_big[..., 0]

        # copying color layers
        color_ab[0, :, :] = img_little[..., 1].reshape(1, self.shape[0], self.shape[1])
        color_ab[1, :, :] = img_little[..., 2].reshape(1, self.shape[0], self.shape[1])

        # loading hist weights
        if self.lossweights is not None:
            weights = self.__getweights__(color_ab)

        return color_ab, grey_little, weights, grey_big, grey_cropped

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

    def dump_results(self, color, grey, gt, nmix, name='result'):
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
        net_result = cv2.cvtColor(net_result, cv2.COLOR_LAB2BGR)
        gt_print = cv2.cvtColor(gt_print, cv2.COLOR_LAB2BGR)

        out_fn_pred = '%s/%s.png' % (self.out_directory, name)
        cv2.imwrite(out_fn_pred, np.concatenate((net_result, border_img, gt_print), axis=1))
        return
