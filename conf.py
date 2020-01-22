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
    'EPOCHS': 3,  # number of epoch to be performed foreach model
    'TRANSF_LEARNING_EPOCH': 3,
    # reload hist weights for the dataset with a quantization coefficient Q_FACTOR
    "TEST_ON_TRAIN_RATE": 5,  # do test every X train epochs
    'BATCHSIZE': 32,
    'TEST_BATCHSIZE': 16,
    # weight stuff
    'RELOAD_WEIGHTS': False,
    'WEIGHT_FILENAME': 'delta_2_test.npy',
    'Q_FACTOR': 26,
    'DELTA_GAUSSIAN': 2,
    'LAMDA': 0.7,
    'NMIX': 3,  # number of samples AKA different colorizations for a given image
    'HIDDEN_SIZE': 128,

    #########
    # CVAE PARAMS

    'TRAIN_CVAE': True,
    'TEST_CVAE': True,
    'CVAE_EPOCH_CHECKPOINT': 0,
    'LOAD_CVAE': False,
    'CVAE_LR': 1e-05,
    'PREDICTION_LR': 5e-04,
    'WARM_UP_TH': 1e-3,
    'CLIP_TH': 20,

    # SCHEDULER PARAMS
    'SCHED_VAE_STEP': 5,
    'SCHED_VAE_GAMMA': 0.1,
    'SCHED_MDN_STEP': 5,
    'SCHED_MDN_GAMMA': 0.1,
    # END OF SCHEDULER PARAMS

    ########
    # OTHER GENERIC PARAMS

    # names of dataset files
    'BIG_EARTH_CVS_NAME':
        '/nas/softechict-nas-2/svincenzi/BigEarth_torch_version/big_earth_all_torch_labels.csv'
    if deploy else
        '/home/potpov/Projects/PycharmProjects/trainship/pytorch_divcolor/datasets/bigearth/csv/big_earth_50000.csv',


    # experiment results dir
    'OUT_DIR': '/nas/softechict-nas-2/mcipriano/experiments' if deploy else 'tests/',
    'TENSORBOARD_DIR': '/nas/softechict-nas-2/mcipriano/' if deploy else '.',
    'SEED': 42,

    "TEST_SPLIT": 0.2,  # train / dataset %
    'NTHREADS': 8,  # data-loader workers

    # "BANDS": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
    "BANDS": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # original image size
    'IMG_W': 128,
    'IMG_H': 128,
    # scaled image size
    'UP_W': 64,
    'UP_H': 64,

    "CLASSES": [
        'Continuous urban fabric', 'Discontinuous urban fabric',
        'Industrial or commercial units', 'Road and rail networks and associated land',
        'Port areas', 'Airports', 'Mineral extraction sites', 'Dump sites',
        'Construction sites', 'Green urban areas',
        'Sport and leisure facilities', 'Non-irrigated arable land',
        'Permanently irrigated land', 'Rice fields', 'Vineyards',
        'Fruit trees and berry plantations', 'Olive groves',
        'Pastures', 'Annual crops associated with permanent crops',
        'Complex cultivation patterns',
        'Land principally occupied by agriculture, with significant areas of natural vegetation',
        'Agro-forestry areas', 'Broad-leaved forest', 'Coniferous forest', 'Mixed forest',
        'Natural grassland', 'Moors and heathland', 'Sclerophyllous vegetation',
        'Transitional woodland/shrub', 'Beaches, dunes, sands', 'Bare rock',
        'Sparsely vegetated areas', 'Burnt areas', 'Glaciers and perpetual snow',
        'Inland marshes', 'Peatbogs', 'Salt marshes', 'Salines', 'Intertidal flats',
        'Water courses', 'Water bodies', 'Coastal lagoons', 'Estuaries', 'Sea and ocean',
    ]
}
