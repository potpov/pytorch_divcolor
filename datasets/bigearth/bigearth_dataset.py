from pathlib import Path
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn as nn
import torch
import numpy as np
import cv2
import time
import os

# "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


# rgb --> CIELab conversion, with normalization [0-1]
def rgb2lab(rgb):
    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab_img[:, :, 0] *= 255 / 100
    lab_img[:, :, 1] += 128
    lab_img[:, :, 2] += 128
    # lab_img /= 255
    lab_img = (lab_img / 128) - 1
    return lab_img[:, :, 1:], lab_img[:, :, 0]


class BigEarthDataset(Dataset):
    def __init__(self, csv_path, random_seed, bands, weights_file=None, n_samples=100000, dsize=128, RGB=1, mode='classification', skip_weights=False):
        """
        Args:
            csv_path (string): path to csv file containing folder name that contain images
        """
        # saving the current dir of this subfolder for local imports
        self.curr_dir = Path(__file__).parent
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the folder paths
        self.folder_path = self.data_info.iloc[:n_samples, 0].tolist()
        # Second column contains the text labels
        self.labels_name = self.data_info.iloc[:n_samples, 1].tolist()
        # Third column contains the number labels
        self.labels_class = self.data_info.iloc[:n_samples, 2].tolist()
        # shuffle the entries, specify the seed
        tmp_shuffle = list(zip(self.folder_path, self.labels_name, self.labels_class))
        np.random.seed(random_seed)
        np.random.shuffle(tmp_shuffle)
        self.folder_path, self.labels_name, self.labels_class = zip(*tmp_shuffle)
        # Calculate len
        self.data_len = len(self.data_info.iloc[:n_samples])
        print("Dataset len: ", self.data_len)
        # bands
        self.bands = torch.BoolTensor(bands)
        # image size
        self.dsize = dsize
        # flag for RGB, to invert the channels
        self.RGB = RGB
        # flag for dataset mode: classification or colorization
        self.mode = mode
        # flag for weights
        self.skip_weights = skip_weights
        if not skip_weights:
            self.binedges = np.load(self.curr_dir / 'pot_weights/ab_quantize.npy')
            self.weights = np.load(os.path.join(self.curr_dir, 'pot_weights', weights_file))

    def __getitem__(self, index):
        # obtain the right folder
        imgs_file = self.folder_path[index][2:-2]  # to remove [\ \]
        # load torch images
        spectral_img = torch.load(imgs_file + '/all_bands.pt')
        # resize the image as specified in the params dsize
        # spectral_img = torch.squeeze(nn.functional.interpolate(input=torch.unsqueeze(spectral_img, dim=0), size=self.dsize))
        spectral_img = torch.squeeze(nn.functional.interpolate(spectral_img, size=self.dsize))

        # take only the bands specified in the init
        spectral_img = spectral_img[self.bands]
        # if RGB: invert the indices as it is saved as BGR
        if self.RGB:
            spectral_img = torch.flip(spectral_img, [0])
        # MODE: MULTILABEL CLASSIFICATION
        if self.mode == "classification":
            # create multi-hot labels vector
            labels_index = list(map(int, self.labels_class[index][1:-1].split(',')))
            labels_class = np.zeros(44)
            labels_class[labels_index] = 1
            # create labels vector, padded with -1
            pad = np.array([-1] * 6)
            pad[:len(labels_index)] = labels_index[:6]
            return spectral_img, torch.tensor(labels_class), torch.tensor(pad)
        else:  # MODE: COLORIZATION
            if self.RGB:
                return self.generate_inputs(spectral_img)
            else:
                indices = torch.tensor([3, 2, 1])
                rgb = torch.index_select(input=spectral_img, dim=0, index=indices)
                return spectral_img, self.generate_inputs(rgb)

    def __len__(self):
        return self.data_len

    def split_dataset(self, thresh_test, thresh_val=None):
        """
        :param thresh: threshold for splitting the dataset in training and test set
        ----for the moment I don't implement the k-fold version----
        :return: the two split (or three, when and if I add the validation set)
        """
        indices = list(range(self.data_len))
        if thresh_val is not None:
            split_val = int(np.floor(thresh_val * self.data_len))
        split_test = int(np.floor(thresh_test * self.data_len))
        # return the train and test index
        return indices[split_val:], indices[split_test:split_val], indices[:split_test]

    def generate_inputs(self, rgb):
        """
        return set of images for the models
        :param rgb: rgb image to be converted
        :return: color_ab: AB color channels, shape 2x64x64
        :return: grey: greyscale image, shape 1x64x64
        :return: weights: set of weights for the color space
        """
        # convert tensor to numpy
        rgb = np.transpose(rgb.numpy(), (1, 2, 0))
        # converting original image to CIELAB --> ps. uso la mia funzione perche' ho i tensori in float32
        img_ab, img_L = rgb2lab(rgb)
        # transpose channels
        img_ab = np.transpose(img_ab, (2, 0, 1))
        # add channels: axis 0
        img_L = np.expand_dims(img_L, axis=0)
        # load weights or return array of ones
        if not self.skip_weights:
            weights = self.__getweights__(img_ab)
        else:
            weights = np.ones_like(img_ab)
        return img_ab, img_L, weights

    def __getweights__(self, img):
        """
        foreach pixel search the bin and get the weight value for that bin in the weights histogram
        :param img: AB channel image
        :return: AB image shape with a weight on each pixel
        """
        # flat the img vector and select A channel + strech it to [-128, 128]
        img_vec = img.reshape(-1)
        img_vec = img_vec * 128.
        img_vec_a = img_vec[:np.prod((self.dsize, self.dsize))]

        # compute min distance between each pixel and the A edges, scale values on Quantization factor
        binid_a = [np.abs(self.binedges - v).argmin() for v in img_vec_a]

        # flat the img vector and select B channel
        img_vec_b = img_vec[np.prod((self.dsize, self.dsize)):]

        # compute min distance between each pixel and the B edges
        binid_b = [np.abs(self.binedges - v).argmin() for v in img_vec_b]

        # merge the min indexes and get values from the hist
        binweights = np.array([self.weights[int(v1), int(v2)] for v1, v2 in zip(binid_a, binid_b)])

        # saving result for this image
        result = np.zeros(img.shape, dtype='f')
        result[0, :, :] = binweights.reshape((self.dsize, self.dsize))
        result[1, :, :] = binweights.reshape((self.dsize, self.dsize))
        return result


if __name__ == "__main__":
    # Call dataset
    big_earth = BigEarthDataset(csv_path='big_earth_all_torch_labels.csv', random_seed=19,
                                bands=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                n_samples=500, dsize=128, RGB=0, mode="colorization", skip_weights=False)
    # train_idx, val_idx, test_idx = big_earth.split_dataset(0.2, 0.4)
    #
    # train_sampler = SubsetRandomSampler(train_idx)
    # test_sampler = SubsetRandomSampler(test_idx)
    seq_sampler = torch.utils.data.SequentialSampler(big_earth)

    # train_loader = torch.utils.data.DataLoader(big_earth, batch_size=1,
    #                                            sampler=train_sampler, num_workers=0)
    # test_loader = torch.utils.data.DataLoader(big_earth, batch_size=1,
    #                                           sampler=test_sampler, num_workers=0)

    train_loader = torch.utils.data.DataLoader(big_earth, batch_size=4,
                                               sampler=seq_sampler, num_workers=0)

    start_time = time.time()
    print("Start time: ", start_time)

    for idx, (img, labels_class, lab_idx) in enumerate(train_loader):
        print(idx)
        print(img.shape)

    print("time: ", time.time() - start_time)



