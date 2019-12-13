import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np
import cv2
import glob
from datasets.bigearth.dict_to_something import load_dict_from_json
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
def custom_loader(path, band, quantiles, size_w, size_h):
    # read the band as it is, with IMREAD_UNCHANGED
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, dsize=(size_w, size_h), interpolation=cv2.INTER_CUBIC)
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
    def __init__(self, conf, csv_path, quantiles, random_seed, writer=None, skip_weights=False):
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
        self.bands = conf['BANDS']

        # configuration file
        self.conf = conf

        # histogram weights
        self.skip_weights = skip_weights
        if not skip_weights:
            self.binedges = np.load(os.path.join(self.curr_dir, 'pot_weights/ab_quantize.npy'))
            self.weights = np.load(os.path.join(self.curr_dir, 'pot_weights', self.conf['WEIGHT_FILENAME']))
            if writer is not None:
                writer.add_histogram(self.conf['WEIGHT_FILENAME'], self.weights, 0)  # plot weights

    def __getitem__(self, index):
        # obtain the right folder
        imgs_file = self.folder_path[index]
        imgs_bands = []
        for b in self.bands:
            for filename in glob.iglob(os.path.join(self.curr_dir, imgs_file)+"/*" + b + ".tif"):
                band = custom_loader(
                    os.path.join(self.curr_dir, filename),
                    b,
                    self.quantiles,
                    self.conf['IMG_W'],
                    self.conf['IMG_H']
                )
                imgs_bands.append(band)
        if len(self.bands) == 3:
            rgb = np.concatenate(imgs_bands[::-1], axis=2)  # inverse order for rgb
            return self.generate_inputs(rgb)
        else:
            spectral_img = np.concatenate(imgs_bands, axis=2)
            spectral_bands, rgb = split_bands(spectral_img)
            return spectral_bands, self.generate_inputs(rgb)

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
        :return: grey: greyscale image, shape 1x64x64
        :return: weights: set of weights for the color space
        :return: grey_big: greyscale image, shape 1x256x256
        :return: grey_cropped: greyscale cropped image, shape 1x224x224
        """

        # converting original image to CIELAB
        img_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        img_lab = ((img_lab * 2.) / 255.) - 1.

        # GREY cropped for MDN
        grey_cropped = np.zeros((1, 224, 224), dtype='f')
        img_cropped = cv2.resize(img_lab, (224, 224))
        grey_cropped[0, :, :] = img_cropped[..., 0]

        # GREY scale image natural shape
        grey = np.zeros((1, self.conf['IMG_W'], self.conf['IMG_H']), dtype='f')
        grey[0, :, :] = img_lab[..., 0]

        # AB channels natural shape
        color_ab = np.zeros((2, self.conf['IMG_W'], self.conf['IMG_H']), dtype='f')
        color_ab[0, :, :] = img_lab[..., 1].reshape(1, self.conf['IMG_W'], self.conf['IMG_H'])
        color_ab[1, :, :] = img_lab[..., 2].reshape(1, self.conf['IMG_W'], self.conf['IMG_H'])

        # load weights or return array of ones
        weights = np.ones((2, self.conf['IMG_W'], self.conf['IMG_H']), dtype='f')
        if not self.skip_weights:
            weights = self.__getweights__(color_ab)

        return color_ab, grey, weights, grey_cropped

    def __getweights__(self, img):
        """
        foreach pixel search the bin and get the weight value for that bin in the weights histogram
        :param img: AB channel image
        :return: AB image shape with a weight on each pixel
        """
        # flat the img vector and select A channel + strech it to [-128, 128]
        img_vec = img.reshape(-1)
        img_vec = img_vec * 128.
        img_vec_a = img_vec[:np.prod((self.conf['IMG_W'], self.conf['IMG_H']))]

        # compute min distance between each pixel and the A edges, scale values on Quantization factor
        binid_a = [np.abs(self.binedges - v).argmin() for v in img_vec_a]

        # flat the img vector and select B channel
        img_vec_b = img_vec[np.prod((self.conf['IMG_W'], self.conf['IMG_H'])):]

        # compute min distance between each pixel and the B edges
        binid_b = [np.abs(self.binedges - v).argmin() for v in img_vec_b]

        # merge the min indexes and get values from the hist
        binweights = np.array([self.weights[int(v1), int(v2)] for v1, v2 in zip(binid_a, binid_b)])

        # saving result for this image
        result = np.zeros(img.shape, dtype='f')
        result[0, :, :] = binweights.reshape((self.conf['IMG_W'], self.conf['IMG_H']))
        result[1, :, :] = binweights.reshape((self.conf['IMG_W'], self.conf['IMG_H']))
        return result