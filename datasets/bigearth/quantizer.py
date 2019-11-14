from .bigearth_dataset import BigEarthDataset
from torch.utils.data import SubsetRandomSampler
import torch
from tqdm import tqdm
import numpy as np
import os


big_earth = BigEarthDataset('big_earth_3000.csv', 'quantiles_3000.json', 42)
train_idx, test_idx = big_earth.split_dataset(0.2)
# define the sampler
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
# define the loader
loader = torch.utils.data.DataLoader(big_earth, batch_size=32,
                                           sampler=train_sampler, num_workers=8)

data_loader = torch.utils.data.DataLoader(
                    big_earth,
                    batch_size=32,
                    sampler=train_sampler,
                    num_workers=8,
                    drop_last=True
                )

curr_dir = os.path.dirname(__file__)
binedges = np.load(os.path.join(curr_dir, 'zhang_weights/ab_quantize.npy')).reshape(2, 313)
weights = {}

for batch_idx, (color_ab, _, _, _, _) in \
        tqdm(enumerate(data_loader), total=len(data_loader)):

    img_weights = np.zeros(color_ab.shape, dtype='f')
    # unroll image and stretch it to [-128, 128]
    img_vec = color_ab.reshape(-1)
    img_vec = img_vec * 128.

    # extracting A channel and its bin edges
    img_vec_a = img_vec[:np.prod((64, 64))]
    binedges_a = binedges[0, ...].reshape(-1)
    # finding the closest bucket
    binid_a = [binedges_a.flat[np.abs(binedges_a - v).argmin()] for v in img_vec_a]
    # extracting B channel and its bin edges
    img_vec_b = img_vec[np.prod((64, 64)):]
    binedges_b = binedges[1, ...].reshape(-1)
    # finding the closest bucket
    binid_b = [binedges_b.flat[np.abs(binedges_b - v).argmin()] for v in img_vec_b]

    # updating weights for the closest buckets
    for v1, v2 in zip(binid_a, binid_b):
        img_weights[v1, v2] = img_weights[v1, v2] + 1
