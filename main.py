from __future__ import print_function
from mdn import Mdn
from cvae import Cvae
from utilities import Utilities
import time
import argparse
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description="colorization model")
parser.add_argument("-load_dir", default=None, help="name of folder where reload experiment")
args = parser.parse_args()

if __name__ == '__main__':

    start_time = time.time()

    # tensor-board log
    writer = SummaryWriter()

    # create or load experiment settings
    utilities = Utilities(args.load_dir)

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
    print("training completed in {} hours".format(round((time.time() - start_time) / 3600), 2))
