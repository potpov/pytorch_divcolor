import socket

# automatically detect if the script is on localhost or on remote server
# in order to fix some path
deploy = True
if socket.gethostname() == 'parrot':
    deploy = False


default_conf = {

    #########
    # MAIN GENERIC PARAMS

    'DATASET_NAME': 'bigearth',  # dataset type for the experiment: bigearth, lfw, ...
    'EPOCHS': 2,  # number of epoch to be performed foreach model
    # reload hist weights for the dataset with a quantization coefficient Q_FACTOR
    'BATCHSIZE': 64,
    'TEST_BATCHSIZE': 12,
    # weight stuff
    'RELOAD_WEIGHTS': True,
    'WEIGHT_FILENAME': 'delta_2_prior.npy',
    'Q_FACTOR': 26,
    'DELTA_GAUSSIAN': 2,
    'LAMDA': 0.5,
    'NMIX': 4,  # number of samples AKA different colorizations for a given image

    #########
    # CVAE PARAMS

    'TRAIN_CVAE': True,
    'TEST_CVAE': True,
    'CVAE_EPOCH_CHECKPOINT': 0,
    'LOAD_CVAE': False,
    'CVAE_LR': 5e-5,

    ########
    # MDN PARAMS

    'TRAIN_MDN': True,  # if train of mdn has to be performed
    'TRAIN_VAE': True,  # if train of vae has to be performed
    'TEST_MDN_VAE': True,

    'VAE_LR': 5e-5,  # VAE learning rate
    'MDN_LR': 1e-3,  # MDN learning rate

    'VAE_EPOCH_CHECKPOINT': 0,
    'MDN_EPOCH_CHECKPOINT': 0,
    'LOAD_MDN': False,
    'LOAD_VAE': False,

    # weights for MDN-VAE losses
    'KL_W': 1e-2,  # kl divergence
    'MAH_W': 0,  # mah loss -> PCA over the most important components 1e-2
    'HIST_W': 1,  # hist loss -> see colorful colorizations
    'GRA_W': 1e1,  # gradient loss using sobel

    # PCA info for the MAH loss
    'PCA_DIR': 'assets/pcomp/lfw',
    'PCA_COMP_NUMBER': 20,

    # SCHEDULER PARAMS
    'SCHED_VAE_STEP': 5,
    'SCHED_VAE_GAMMA': 0.1,
    'SCHED_MDN_STEP': 5,
    'SCHED_MDN_GAMMA': 0.1,
    # END OF SCHEDULER PARAMS

    ########
    # OTHER GENERIC PARAMS

    # names of dataset files
    'BIG_EARTH_CVS_NAME': 'big_earth_50000.csv' if deploy else 'big_earth_3000.csv',
    'BIG_EARTH_QNTL_NAME': 'quantiles_50000.json' if deploy else 'quantiles_3000.json',

    'OUT_DIR': 'tests/',  # experiment results dir
    'SEED': 42,

    "TEST_SPLIT": 0.2,  # train / dataset %
    'NTHREADS': 8,  # data-loader workers
    'HIDDENSIZE': 64,  # encoder and mdn output size

    "BANDS": ["B02", "B03", "B04"],
    # original image size
    'IMG_W': 64,
    'IMG_H': 64,
    # scaled image size
    'UP_W': 64,
    'UP_H': 64,
}
