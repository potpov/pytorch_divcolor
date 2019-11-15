from bigearth_dataset import BigEarthDataset
from torch.utils.data import SubsetRandomSampler
import torch
from tqdm import tqdm
import numpy as np
import os
import cv2


big_earth = BigEarthDataset('big_earth_3000.csv', 'quantiles_3000.json', 42)
train_idx, test_idx = big_earth.split_dataset(0.2)
# define the sampler
train_sampler = SubsetRandomSampler(train_idx)
test_sampler = SubsetRandomSampler(test_idx)
# define the loader
loader = torch.utils.data.DataLoader(big_earth, batch_size=1,
                                           sampler=train_sampler, num_workers=8)

data_loader = torch.utils.data.DataLoader(
                    big_earth,
                    batch_size=1,
                    sampler=train_sampler,
                    num_workers=8,
                    drop_last=True
                )

curr_dir = os.path.dirname(__file__)
binedges = np.load(os.path.join(curr_dir, 'zhang_weights/ab_quantize.npy')).reshape(2, 313)
weights = np.zeros((210, 210), dtype='f')
i = 0

for batch_idx, (color_ab, _, _, _, _) in \
        tqdm(enumerate(data_loader), total=len(data_loader)):

    img_weights = np.zeros((210, 210), dtype='f')  # possible bins indexes are from -110 to 110
    # unroll image and stretch it to [-128, 128]
    img_vec = color_ab.squeeze(0).numpy().reshape(-1)
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

    # updating weights for the closest buckets of this image
    for v1, v2 in zip(binid_a, binid_b):
        img_weights[v1.astype('int8'), v2.astype('int8')] = img_weights[v1.astype('int8'), v2.astype('int8')] + 1

    # convert occurrences in probabilities
    norm_img_weights = img_weights / np.sum(img_weights)
    # saving this image block of probabilities
    i = i + 1
    weights = weights + norm_img_weights

# mean over all the weights of all the images
weights = weights / i
# saving only the values in binedges like the indian guy did
countbins = np.empty(313, dtype='f')
for i in range(313):
    countbins[i] = weights[int(binedges[0, i]), int(binedges[1, i])]

# assigning negligible value to nevermet-values to avoid division by zero
# this value won't affect our losses as the get_weights function is linked to the
# grantruth colors, so if a quantized value is never met its weight will never
# be loaded from the __get_weights__ function in the dataloader
countbins[countbins == 0] = 1e-09

# saving result probabilities
np.save('pot_weights/prior_bigearth.npy', countbins)
