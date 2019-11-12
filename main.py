from __future__ import print_function
import models
from utilities import Utilities

if __name__ == '__main__':

    # None for new experiment, string or list of strings of folders to load experiments
    config_dirs = {
        'bigearth': '2019-11-12_18:25:41.518313',
        # 'lfw': '2019-11-12_15:45:51.574263',
    }
    # config_dirs = None

    if config_dirs is not None and len(config_dirs) > 0:
        # load experiments
        for dataset, dir in config_dirs.items():
            print("starting experiment for file {}".format(dir))
            utilities = Utilities(dir, dataset)
            models.model(utilities)
    # one config or new experiment
    else:
        utilities = Utilities(config_dirs)
        models.model(utilities)
