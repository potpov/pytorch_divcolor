from __future__ import print_function
from mdn import Mdn
from cvae import Cvae
from utilities import Utilities
import time
import argparse
from tensorboardX import SummaryWriter
import pathlib


parser = argparse.ArgumentParser(description="colorization model")
parser.add_argument('-l', '--list', help='test dir list inside quotes, separated with space: "test1 test2 test3"', type=str)
args = parser.parse_args()


def train(utilities, exp_title):

    # tensor-board log
    logs_dir = pathlib.Path(__file__).parent / 'runs' / '{}'.format(exp_title)
    logs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logs_dir)

    # reload hist weights?
    if utilities.conf['RELOAD_WEIGHTS']:
        utilities.reload_weights()

    train_loader = utilities.load_data('train')
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
        cvae.train(train_loader, writer)
        cvae.check_train(train_loader, writer)
    if utilities.conf['TEST_CVAE']:
        cvae.test(test_loader, writer)

    utilities.test_complete()


if __name__ == '__main__':

    start_time = time.time()

    # if no dir was provided create new experiment with default parameters
    if not args.list:
        utilities = Utilities(None)
        train(utilities, exp_title=utilities.datalog)
        print("training completed in {} hours".format(round((time.time() - start_time) / 3600), 2))
    else:
        # foreach dir with config file launch experiment
        experiments = [str(dir_list) for dir_list in args.list.split(' ')]
        for experiment in experiments:
            print("loading experiment with name: ", experiment)
            # create or load experiment settings
            utils = Utilities(experiment)
            train(utils, exp_title=experiment)
            print("training completed in {} hours".format(round((time.time() - start_time) / 3600), 2))



