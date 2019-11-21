from __future__ import print_function
import models
from utilities import Utilities
import time

if __name__ == '__main__':

    start_time = time.time()
    # None for new experiment, string or list of strings of folders to load experiments
    config_dirs = {
        'bigearth': 'cvae'
    }

    if len(config_dirs) > 0:
        # load experiments
        for dataset, dir in config_dirs.items():
            print("starting experiment for file {}".format(dir))
            utilities = Utilities(dir, dataset)
            models.model(utilities)
            print("training completed in {} hours".format(round((time.time() - start_time) / 3600), 2))
    # one config or new experiment
    else:
        utilities = Utilities(config_dirs)
        models.model(utilities)
        print("training completed in {} hours".format(round((time.time() - start_time) / 3600), 2))
