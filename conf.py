
default_conf = {

    'DATASET_NAME': 'bigearth',  # bigearth, lfw, ...

    'TRAIN_MDN': True,  # if train of mdn has to be performed
    'TRAIN_VAE': True,  # if train of vae has to be performed
    'LOAD_MDN': False,  # if existing weights has to be loaded
    'LOAD_VAE': False,  # same as above
    'TEST_MDN_VAE': True,

    'LOAD_CVAE': False,
    'TRAIN_CVAE': True,
    'TEST_CVAE': True,

    'OUT_DIR': 'tests/',

    'SEED': 42,
    "TEST_SPLIT": 0.2,

    'EPOCHS': 20,
    'NTHREADS': 8,  # data-loader workers
    'HIDDENSIZE': 64,  # encoder and mdn output size
    'NMIX': 8,  # number of samples AKA different colorizations for a given image
    'BATCHSIZE': 32,


    'VAE_LR': 5e-5,  # VAE learning rate
    'MDN_LR': 1e-3,  # MDN learning rate
    'SCHED_VAE_STEP': 5,
    'SCHED_VAE_GAMMA': 0.1,
    'SCHED_MDN_STEP': 5,
    'SCHED_MDN_GAMMA': 0.1,

    'PCA_DIR': 'assets/pcomp/lfw',
    'PCA_COMP_NUMBER': 20,

    # weights for the losses
    'KL_W': 1e-2,  # starts around e4     1e-2
    'MAH_W': 1e-2,  # starts around e6     1e-2
    'HIST_W': 1,   # starts around 1
    'GRA_W': 1e1,  # starts around e-2   1e1

    'IMG_W': 64,
    'IMG_H': 64,
    'SCALED_W': 64,
    'SCALED_H': 64,
}
