from __future__ import print_function
import models
from utilities import Utilities

if __name__ == '__main__':
    # None for new experiment, string or list of strings of folder to load experiments
    config_dirs = [
        '2019-11-08_12:12:18',
        '2019-11-08_12:12:17'
    ]

    if len(config_dirs) > 0:
        # load experiments
        for config_dir in config_dirs:
            print("starting experiment for file {}".format(config_dir))
            utilities = Utilities(config_dir)
            models.model(utilities)
    # one config or new experiment
    else:
        utilities = Utilities(config_dirs)
        models.model(utilities)
