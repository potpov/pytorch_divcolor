from __future__ import print_function
import models
from utilities import Utilities

if __name__ == '__main__':

    # None for new experiment, string or list of strings of folders to load experiments
    config_dirs = {
        # 'bigearth': '3'
    }

    if len(config_dirs) > 0:
        # load experiments
        for dataset, dir in config_dirs.items():
            print("starting experiment for file {}".format(dir))
            utilities = Utilities(dir, dataset)
            models.model(utilities)
    # one config or new experiment
    else:
        utilities = Utilities(config_dirs)
        models.model(utilities)
