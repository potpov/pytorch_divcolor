from __future__ import print_function
from cvae import Cvae
from utilities import Utilities
import time
import argparse
from tensorboardX import SummaryWriter
import os


os.environ['OMP_NUM_THREADS'] = "1"
os.environ['KML_NUM_THREADS'] = "1"

parser = argparse.ArgumentParser(description="colorization model")
parser.add_argument(
    '-e', '--experiments',
    help='name of new/previous tests. space separated: "test1 test2 test3"',
    required=True,
)
parser.add_argument('--overwrite', help='overwrite existing folders', action='store_true')
args = parser.parse_args()


def colorization(utilities):

    ########
    # tensor-board log

    logs_dir = os.path.join(utilities.conf['TENSORBOARD_DIR'], 'runs', '{}'.format(utilities.title))
    writer = SummaryWriter(logs_dir, purge_step=0)

    #####
    # reload hist weights?
    if utilities.conf['RELOAD_WEIGHTS']:
        utilities.reload_weights()

    train_loader = utilities.load_data('train', mode='colorization', rgb=False, writer=writer)
    test_loader = utilities.load_data('test', mode='colorization', rgb=False,)

    ###########
    # CVAE

    cvae = Cvae(utilities)
    if utilities.conf['LOAD_CVAE']:
        cvae.load_weights()
    if utilities.conf['TRAIN_CVAE']:
        cvae.train(train_loader, test_loader, writer)
    if utilities.conf['TEST_CVAE']:
        print("starting final testing")
        cvae.test(test_loader, writer)

    # FEATURE EXTRACTION
    pred_train_loader = utilities.load_data('prediction', mode='classification', rgb=False)
    pred_test_loader = utilities.load_data('prediction', mode='classification', rgb=False)

    cvae.transfer_learning_train(pred_train_loader, writer)
    cvae.transfer_learning_test(pred_test_loader, writer)

    utilities.test_complete()


if __name__ == '__main__':

    start_time = time.time()

    # foreach dir with config file launch experiment else create new experiment
    experiments = [str(exp) for exp in args.experiments.split(' ')]
    for experiment in experiments:
        # create or load experiment settings
        utils = Utilities(experiment, overwrite=args.overwrite)
        colorization(utils)
        print("training completed in {} hours".format(round((time.time() - start_time) / 3600), 2))



