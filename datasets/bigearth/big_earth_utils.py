import csv
import time
from pathlib import Path
import glob
import cv2
import numpy as np
import pandas as pd
from distutils.dir_util import copy_tree
from dict_to_something import save_dict_to_json
import os


def big_earth_to_csv(big_e_path, num_samples, csv_filename):
    path = Path(big_e_path)
    print("collecting dirs...")
    start_time = time.time()
    # if num_samples == -1 write a csv for the entire dataset
    if num_samples == -1:
        # [str(e)] because csv writer.writerows needs a list of lists..
        dirs = [[str(e)] for e in path.iterdir() if e.is_dir()]
    else:
        # zip and range() are useful for choose only a specific number of example
        dirs = [[str(e)] for _, e in zip(range(num_samples), path.iterdir()) if e.is_dir()]
    # write the dirs on a csv file
    print("writing on csv...")
    with open(csv_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(dirs)
    print("finishing in {}".format(time.time() - start_time))


def copy_dir(big_e_path, small_e_path, num_samples):
    src_path = Path(big_e_path)
    dirs = [e for _, e in zip(range(num_samples), src_path.iterdir()) if e.is_dir()]
    dst_path = Path(small_e_path)
    for d in dirs:
        copy_tree(str(d), str(dst_path))


# function to calculate the min and max quantiles
def min_max_quantile(paths, n_samples=200):
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    quantiles = {}
    for b in bands:
        imgs = []
        for i in range(n_samples):
            path = paths[i] # i choose the i-th path of the list
            for filename in glob.iglob(path + "/*" + b + ".tif"): # this for return one image
                img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                imgs.append(img)
        imgs = np.stack(imgs, axis=0).reshape(-1)
        quantiles[b] = {
            'min_q': np.quantile(imgs, 0.02),
            'max_q': np.quantile(imgs, 0.98)
        }
        print(b, quantiles[b])
    return quantiles


def csv_to_list(csv_path):
    data = pd.read_csv(csv_path, header=None)
    paths = data.iloc[:, 0].tolist()
    return paths


if __name__ == '__main__':
    data_dir = '/nas/softechict-nas-1/mcipriano/'
    big_earth_to_csv(os.path.join(data_dir, "BigEarthNet-v1.0/"), 50000, os.path.join(data_dir, "big_earth_50000.csv"))
    # big_earth_to_csv("/nas/softechict-nas-2/svincenzi/BigEarthNet-v1.0", -1, "big_earth_all.csv")

    paths = csv_to_list('/nas/softechict-nas-1/mcipriano/big_earth_50000.csv')
    # # calculate the min_max quantile for every spectral bands, specifying the number of samples to use
    q = min_max_quantile(paths)
    # # save the result on a json file
    save_dict_to_json(q, os.path.join(data_dir, "/nas/softechict-nas-1/mcipriano/quantiles_50000.json"))
    # copy_dir('/nas/softechict-nas-2/svincenzi/BigEarthNet-v1.0', '/nas/softechict-nas-2/svincenzi/BigEarthNet_small', 300)



############################### function for pad the images with pil....copyMakeBorder() is the cv2 module

# def padding(img,expected_size):
#     desired_size = expected_size
#     delta_width = desired_size - img.size[0]
#     delta_height = desired_size - img.size[1]
#     pad_width = delta_width //2
#     pad_height = delta_height //2
#     padding = (pad_width,pad_height,delta_width-pad_width,delta_height-pad_height)
#     return ImageOps.expand(img, padding)

# resnet = models.resnet18(pretrained=False)
# input = torch.Tensor(1, 3, 60, 60)
#
# feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
# list_resnet = list(resnet.children())[:-1]
#
# # for idx in range(len(list_resnet)):
# #     input = list_resnet[idx](input)
#
# out = feature_extractor(input)
#
# print(out.shape)
