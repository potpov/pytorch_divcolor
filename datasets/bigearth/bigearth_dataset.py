import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import cv2
import glob
from .dict_to_something import load_dict_from_json
import os


# function for calculate the quantile
def quantiles_std(img, band, quantiles):
    min_q = quantiles[band]['min_q']
    max_q = quantiles[band]['max_q']
    img[img < min_q] = min_q
    img[img > max_q] = max_q
    img_dest = np.zeros_like(img)
    img_dest = cv2.normalize(img, img_dest, 0, 255, cv2.NORM_MINMAX)
    return img_dest.astype('uint8')


# function for read, resize and normalize the images
def custom_loader(path, band, quantiles):
    # read the band as it is, with IMREAD_UNCHANGED
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
    img = quantiles_std(img, band, quantiles)
    w, h = img.shape
    return img.reshape(w, h, 1)


def split_bands(spectral_img):
    indices = [0, 4, 5, 6, 7, 8, 9, 10, 11]
    indices_rgb = [3, 2, 1]
    spectral_bands = np.take(spectral_img, indices=indices, axis=2)
    rgb = np.take(spectral_img, indices=indices_rgb, axis=2)
    return spectral_bands, rgb


class BigEarthDataset(Dataset):
    def __init__(self, csv_path, quantiles, random_seed):
        """
        Args:
            csv_path (string): path to csv file containing folder name that contain images
        """
        # saving the current dir of this subfolder for local imports
        self.curr_dir = os.path.dirname(__file__)
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(os.path.join(self.curr_dir, csv_path), header=None)
        # First column contains the folder paths
        self.folder_path = self.data_info.iloc[:, 0].tolist()
        # shuffle the entries, specify the seed
        np.random.seed(random_seed)
        np.random.shuffle(self.folder_path)
        # Calculate len
        self.data_len = len(self.data_info)
        # load quantiles json file
        self.quantiles = load_dict_from_json(os.path.join(self.curr_dir, quantiles))
        # bands
        self.bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

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

    def __getitem__(self, index):
        # obtain the right folder
        imgs_file = self.folder_path[index]
        imgs_bands = []
        for b in self.bands:
            for filename in glob.iglob(os.path.join(self.curr_dir, imgs_file)+"/*" + b + ".tif"):
                band = custom_loader(os.path.join(self.curr_dir, filename), b, self.quantiles)
                imgs_bands.append(band)
        spectral_img = np.concatenate(imgs_bands, axis=2)
        spectral_bands, rgb = split_bands(spectral_img)
        # return self.to_tensor(spectral_bands), self.to_tensor(rgb)
        return self.generate_inputs(rgb)

    def __len__(self):
        return self.data_len

    def split_dataset(self, thresh):
        """
        :param thresh: threshold for splitting the dataset in training and test set
        ----for the moment I don't implement the k-fold version----
        :return: the two split (or three, when and if I add the validation set)
        """
        indices = list(range(self.data_len))
        split = int(np.floor(thresh * self.data_len))
        # return the train and test index
        return indices[split:], indices[:split]

    def generate_inputs(self, rgb):
        """
        return set of images for the models
        :param rgb: rgb image to be converted
        :return: color_ab: AB color channels, shape 2x64x64
        :return: grey_little: greyscale image, shape 1x64x64
        :return: weights: set of weights for the color space
        :return: grey_big: greyscale image, shape 1x256x256
        :return: grey_cropped: greyscale cropped image, shape 1x224x224
        """
        color_ab = np.zeros((2, 64, 64), dtype='f')
        weights = np.ones((2, 64, 64), dtype='f')
        grey_little = np.zeros((1, 64, 64), dtype='f')
        grey_big = np.zeros((1, 256, 256), dtype='f')
        grey_cropped = np.zeros((1, 224, 224), dtype='f')

        # converting original image to CIELAB
        img_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        img_lab = ((img_lab * 2.) / 255.) - 1.

        # creating scaled versions of the image
        img_big = cv2.resize(img_lab, (256, 256))
        img_little = cv2.resize(img_lab, (64, 64))
        img_cropped = cv2.resize(img_lab, (224, 224))

        # copying grey scale layers
        grey_cropped[0, :, :] = img_cropped[..., 0]
        grey_little[0, :, :] = img_little[..., 0]
        grey_big[0, :, :] = img_big[..., 0]

        # copying color layers
        color_ab[0, :, :] = img_little[..., 1].reshape(1, 64, 64)
        color_ab[1, :, :] = img_little[..., 2].reshape(1, 64, 64)

        # load weights
        if self.lossweights is not None:
            weights = self.__getweights__(color_ab)

        return color_ab, grey_little, weights, grey_big, grey_cropped

    def __getweights__(self, img):
        img_vec = img.reshape(-1)
        img_vec = img_vec * 128.
        img_lossweights = np.zeros(img.shape, dtype='f')
        img_vec_a = img_vec[:np.prod((64, 64))]
        binedges_a = self.binedges[0, ...].reshape(-1)
        binid_a = [binedges_a.flat[np.abs(binedges_a - v).argmin()] for v in img_vec_a]
        img_vec_b = img_vec[np.prod((64, 64)):]
        binedges_b = self.binedges[1, ...].reshape(-1)
        binid_b = [binedges_b.flat[np.abs(binedges_b - v).argmin()] for v in img_vec_b]
        binweights = np.array([self.lossweights[v1][v2] for v1, v2 in zip(binid_a, binid_b)])
        img_lossweights[0, :, :] = binweights.reshape((64, 64))
        img_lossweights[1, :, :] = binweights.reshape((64, 64))
        return img_lossweights