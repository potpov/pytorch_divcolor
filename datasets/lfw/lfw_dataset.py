import cv2
import numpy as np
from torch.utils.data import Dataset
import os


class Colordata(Dataset):

    def __init__(self, conf, shape=(64, 64), outshape=(256, 256), split='train'):

        self.img_fns = []
        self.conf = conf
        self.curr_dir = os.path.dirname(__file__)

        with open(os.path.join(self.curr_dir, f'list.{split}.vae.txt'), 'r') as ftr:
            for img_fn in ftr:
                self.img_fns.append(img_fn.strip('\n'))

        self.img_num = len(self.img_fns)
        self.shape = shape
        self.outshape = outshape

        # histogram weights
        self.lossweights = None
        countbins = 1. / np.load(os.path.join(self.curr_dir, 'zhang_weights/prior_probs.npy'))
        binedges = np.load(os.path.join(self.curr_dir, 'zhang_weights/ab_quantize.npy')).reshape(2, 313)
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
        """
        return set of images for the models
        :param idx: batch id
        :return: color_ab: AB color channels, shape 2x64x64
        :return: grey_little: greyscale image, shape 1x64x64
        :return: weights: set of weights for the color space
        :return: grey_big: greyscale image, shape 1x256x256
        :return: grey_cropped: greyscale cropped image, shape 1x224x224
        """
        color_ab = np.zeros((2, self.shape[0], self.shape[1]), dtype='f')
        weights = np.ones((2, self.shape[0], self.shape[1]), dtype='f')
        grey_little = np.zeros((1, self.shape[0], self.shape[1]), dtype='f')
        grey_big = np.zeros((1, self.outshape[0], self.outshape[1]), dtype='f')
        grey_cropped = np.zeros((1, 224, 224), dtype='f')

        # converting original image to CIELAB
        img = cv2.imread(os.path.join(self.curr_dir, self.img_fns[idx]))
        img_norm = img.astype(np.float32)/255
        img_lab = cv2.cvtColor(img_norm, cv2.COLOR_RGB2LAB)
        # img_lab = ((img_lab * 2.) / 255.) - 1.

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
        # if self.lossweights is not None:
        #    weights = self.__getweights__(color_ab)

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
