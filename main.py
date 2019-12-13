from __future__ import print_function
from mdn import Mdn
from cvae import Cvae
from utilities import Utilities
import time
import argparse
from tensorboardX import SummaryWriter
import os
import shutil


os.environ['OMP_NUM_THREADS'] = "1"
os.environ['KML_NUM_THREADS'] = "1"

parser = argparse.ArgumentParser(description="colorization model")
parser.add_argument(
    '-e', '--experiments',
    help='name of new/previous tests. space separated: "test1 test2 test3"',
    required=True,
    type=str
)
args = parser.parse_args()


def train(utilities):

    ########
    # tensor-board log

    logs_dir = os.path.join(utilities.conf['TENSORBOARD_DIR'], 'runs', '{}'.format(utilities.title))
    writer = SummaryWriter(logs_dir, purge_step=0)

    #####
    # reload hist weights?
    if utilities.conf['RELOAD_WEIGHTS']:
        utilities.reload_weights()

    train_loader = utilities.load_data('train', writer)
    test_loader = utilities.load_data('test')

    ###########
    # MDN PART

    mdn = Mdn(utilities)
    if utilities.conf['LOAD_VAE']:
        mdn.load_vae_weights()
    if utilities.conf['TRAIN_VAE']:
        mdn.train_vae(train_loader, writer)
    if utilities.conf['LOAD_MDN']:
        mdn.load_mdn_weights()
    if utilities.conf['TRAIN_MDN']:
        mdn.train_mdn(train_loader, writer)
    if utilities.conf['TEST_MDN_VAE']:
        mdn.test(test_loader, writer)

    ###########
    # CVAE PART

    cvae = Cvae(utilities)
    if utilities.conf['LOAD_CVAE']:
        cvae.load_weights()
    if utilities.conf['TRAIN_CVAE']:
        cvae.train(train_loader, test_loader, writer)
    if utilities.conf['TEST_CVAE']:
        print("starting final testing")
        cvae.test(test_loader, writer)

    utilities.test_complete()


if __name__ == '__main__':

    start_time = time.time()

    # foreach dir with config file launch experiment else create new experiment
    experiments = [str(exp) for exp in args.experiments.split(' ')]
    for experiment in experiments:
        # create or load experiment settings
        utils = Utilities(experiment)
        train(utils)
        print("training completed in {} hours".format(round((time.time() - start_time) / 3600), 2))



