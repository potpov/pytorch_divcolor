import socket

# automatically detect if the script is on localhost or on remote server
# in order to fix some path
deploy = True
if socket.gethostname() == 'parrot':
    deploy = False


default_conf = {

    'DATASET_NAME': 'bigearth',  # dataset type for the experiment: bigearth, lfw, ...
    'EPOCHS': 10,  # number of epoch to be performed foreach model

    'TRAIN_MDN': False,  # if train of mdn has to be performed
    'TRAIN_VAE': False,  # if train of vae has to be performed
    'TRAIN_CVAE': True,

    'TEST_MDN_VAE': False,
    'TEST_CVAE': True,

    # names of dataset files
    'BIG_EARTH_CVS_NAME': 'big_earth_50000.csv' if deploy else 'big_earth_3000.csv',
    'BIG_EARTH_QNTL_NAME': 'quantiles_50000.json' if deploy else 'quantiles_3000.json',

    # following pars are !AUTOMATICALLY UPDATED!
    'VAE_EPOCH_CHECKPOINT': 0,
    'MDN_EPOCH_CHECKPOINT': 0,
    'CVAE_EPOCH_CHECKPOINT': 0,

    'LOAD_MDN': False,
    'LOAD_VAE': False,
    'LOAD_CVAE': False,
    # end of automatically updated pars

    # reload hist weights for the dataset with a quantization coefficient Q_FACTOR
    'RELOAD_WEIGHTS':  True,
    'WEIGHT_FILENAME': 'delta_2_prior.npy',
    'Q_FACTOR': 26,
    'DELTA_GAUSSIAN': 2,

    # experiment results dir
    'OUT_DIR': 'tests/',

    'SEED': 42,
    "TEST_SPLIT": 0.2,  # train / dataset %

    'NTHREADS': 8,  # data-loader workers
    'HIDDENSIZE': 64,  # encoder and mdn output size
    'NMIX': 8,  # number of samples AKA different colorizations for a given image
    'BATCHSIZE': 32,


    'VAE_LR': 5e-5,  # VAE learning rate
    'MDN_LR': 1e-3,  # MDN learning rate
    'CVAE_LR': 5e-5,

    # SCHEDULER PARAMS
    'SCHED_VAE_STEP': 5,
    'SCHED_VAE_GAMMA': 0.1,
    'SCHED_MDN_STEP': 5,
    'SCHED_MDN_GAMMA': 0.1,
    # END OF SCHEDULER PARAMS

    # PCA info for the MAH loss
    'PCA_DIR': 'assets/pcomp/lfw',
    'PCA_COMP_NUMBER': 20,

    # weights for the losses
    'KL_W': 1e-2,  # kl divergence
    'MAH_W': 0,  # mah loss -> PCA over the most important components 1e-2
    'HIST_W': 1,   # hist loss -> see colorful colorizations
    'GRA_W': 1e1,  # gradient loss using sobel

    # original image size
    'IMG_W': 64,
    'IMG_H': 64,
    # scaled image size
    'SCALED_W': 64,
    'SCALED_H': 64,
}
