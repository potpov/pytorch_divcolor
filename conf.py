
OUT_DIR = 'data/output/lfw/'  # where to save images
LISTDIR = 'data/imglist/lfw/'  # list of input image names
FEATSLISTDIR = 'data/featslist/lfw/'  # no idea about this
NTHREADS = 8  # data-loader workers
HIDDENSIZE = 64  # encoder and mdn output size
NMIX = 8  # number of samples AKA different colorizations for a given image
BATCHSIZE = 32
EPOCHS = 15


VAE_LR = 5e-5  # VAE learning rate
MDN_LR = 1e-3  # MDN learning rate
SCHED_VAE_STEP = 5
SCHED_VAE_GAMMA = 0.1
SCHED_MDN_STEP = 5
SCHED_MDN_GAMMA = 0.1

PCA_DIR = 'data/pcomp/lfw'
PCA_COMP_NUMBER = 20

# weights for the losses
KL_W = 1e-2  # starts around e4     1e-2
MAH_W = 1e-2  # starts around e6     1e-2
HIST_W = 1   # starts around 1
GRA_W = 1e1  # starts around e-2   1e1


IMG_W = 64
IMG_H = 64
SCALED_W = 64
SCALED_H = 64
