
OUT_DIR = 'data/output/lfw/'  # where to save images
LISTDIR = 'data/imglist/lfw/'  # list of input image names
FEATSLISTDIR = 'data/featslist/lfw/'  # no idea about this
NTHREADS = 8  # data-loader workers
HIDDENSIZE = 64  # encoder and mdn output size
NMIX = 8  # number of samples AKA different colorizations for a given image
BATCHSIZE = 32
EPOCHS = 20
VAE_LR = 5e-5  # VAE learning rate
MDN_LR = 1e-3  # MDN learning rate

