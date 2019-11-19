from .bigearth_dataset import BigEarthDataset
from torch.utils.data import SubsetRandomSampler
import torch
from tqdm import tqdm
import numpy as np
import cv2
import os


def quantization(conf, q_factor=100, skip_hist=False, skip_edges=False):

    #################################
    # loading dataset without weights

    big_earth = BigEarthDataset(conf['BIG_EARTH_CVS_NAME'], conf['BIG_EARTH_QNTL_NAME'], 42, skip_weights=True)
    idx, _ = big_earth.split_dataset(0)
    sampler = SubsetRandomSampler(idx)

    data_loader = torch.utils.data.DataLoader(
        big_earth, batch_size=32, sampler=sampler, num_workers=8, drop_last=True
    )

    ####################
    # CREATING HISTOGRAM
    curr_dir = os.path.dirname(__file__)

    if not skip_hist:
        print("going to calculate weights. this could take a while.. QFACTOR:" + str(q_factor))
        probs = np.zeros((q_factor, q_factor), dtype='f')
        i = 0

        for batch_idx, (color_ab, _, _, _, _) in \
                tqdm(enumerate(data_loader), total=len(data_loader)):

            # unroll image and stretch it to [-128, 128]
            img_vec = color_ab.squeeze(0).numpy()
            img_vec = img_vec * 128.
            img_vec = np.transpose(img_vec, (1, 2, 3, 0))
            hist = cv2.calcHist(img_vec, [0, 1], None, [q_factor, q_factor], [-128, 128, -128, 128])
            norm_hist = hist / np.sum(hist)
            probs = probs + norm_hist
            i = i + 1

        probs = probs / i
        probs[probs == 0] = 1e-09
        np.save(os.path.join(curr_dir, 'pot_weights/prior_bigearth.npy'), probs)

    ###############
    # CREATING BINS

    if not skip_edges:
        print("creating edges array..")
        binedges = np.ones(q_factor, dtype='f')
        distance = 128*2 / q_factor
        for i in range(0, q_factor):
            binedges[i] = -128 + (distance * i)
        np.save(os.path.join(curr_dir, 'pot_weights/ab_quantize.npy'), binedges)
