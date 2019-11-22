from .bigearth_dataset import BigEarthDataset
from torch.utils.data import SubsetRandomSampler
import torch
from tqdm import tqdm
import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter


def quantization(conf, q_factor=100, skip_hist=False):

    #################################
    # loading dataset without weights

    big_earth = BigEarthDataset(conf, conf['BIG_EARTH_CVS_NAME'], conf['BIG_EARTH_QNTL_NAME'], 42, skip_weights=True)
    idx, _ = big_earth.split_dataset(0)
    sampler = SubsetRandomSampler(idx)

    data_loader = torch.utils.data.DataLoader(
        big_earth, batch_size=32, sampler=sampler, num_workers=8, drop_last=False
    )

    curr_dir = os.path.dirname(__file__)

    ###############
    # CREATING BINS
    binedges = np.ones(q_factor + 1, dtype='f')
    distance = 128 * 2 / q_factor
    for i in range(0, q_factor + 1):
        binedges[i] = -128 + (distance * i)
    np.save(os.path.join(curr_dir, 'pot_weights/ab_quantize.npy'), binedges)

    ####################
    # CREATING HISTOGRAM

    if not skip_hist:
        print("going to calculate weights. this could take a while.. QFACTOR:" + str(q_factor))
        tot_hist = np.zeros((q_factor, q_factor), dtype='f')

        for batch_idx, (color_ab, _, _, _, _) in \
                tqdm(enumerate(data_loader), total=len(data_loader)):

            # unroll image and stretch it to [-128, 128]
            img_vec = color_ab.numpy() * 128.
            img_vec = np.transpose(img_vec, (0, 2, 3, 1))
            img_vec = np.reshape(img_vec, (-1, 2))
            img_a_ch, img_b_ch = img_vec[:, 0], img_vec[:, 1]
            hist = np.histogram2d(img_a_ch, img_b_ch, bins=binedges)[0]
            tot_hist = tot_hist + hist

        # hist to probs
        probs = tot_hist / np.sum(tot_hist)
        # smoothing probs
        probs_gauss = gaussian_filter(probs, sigma=conf['DELTA_GAUSSIAN'])
        # blend with uniform prob
        uniform = np.ones_like(probs, dtype=probs.dtype)*(1/(probs.shape[0]**2))
        probs = conf['LAMDA'] * probs_gauss + (1-conf['LAMDA']) * uniform
        # calculate and norm weights
        weights = 1/probs
        w_avg = np.sum(probs_gauss * weights)
        # normalizing weights between [a, b] where a=0.1
        # b is automatically calculated in order to have
        # sum(prob*weights) equal to 1
        b = np.add(
            (np.max(weights) - np.min(weights)) / (w_avg - np.min(weights)),
            -0.1 * (np.max(weights)-w_avg) / (w_avg - np.min(weights))
        )
        # final normalization
        new_weights = 0.1 + ((weights - np.min(weights)) * (b - 0.1)) / (np.max(weights)-np.min(weights))
        # saving weights for dataloader
        np.save(os.path.join(curr_dir, 'pot_weights', conf['WEIGHT_FILENAME']), new_weights)
