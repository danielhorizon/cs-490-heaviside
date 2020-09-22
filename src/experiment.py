from setup_paths import *
try:
   import cPickle as pickle
except:
   import pickle

import logging


class Experiment():
    # keep track of our models for this trial, as random initialization matters
    def __init__(self, args, pkl_path=None):
        if pkl_path is not None:
            self.pkl = pkl_path
            self.load()
        else:
            self.args = args

            # all indexed by dataset name, then loss (where applicable)
            self.models = {}
            self.histories = {}
            self.results = {}
            self.predictions = {}

            self.path = self.makepath()
            self.tensorboard_log_path = self.path.joinpath("..", "tensorboard")

        self.get_logger()
        self.log.info(self.args)

    def get_logger(self):
        self.log = logging.getLogger(self.args.prefix)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(self.path.joinpath("experiment.log"))
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.ERROR)

        # Create formatters and add it to handlers
        c_format = logging.Formatter("%(asctime)s [%(levelname)s][%(threadName)s] %(name)s: %(message)s")
        f_format = logging.Formatter("%(asctime)s [%(levelname)s][%(threadName)s] %(name)s: %(message)s")
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.log.addHandler(c_handler)
        self.log.addHandler(f_handler)

    def makepath(self):
        experiment_name = "_".join([
            self.args.prefix,
            self.args.dataset,
            self.args.loss
        ])
        path = ROOT_PATH.joinpath(self.args.output_folder).joinpath(self.args.experiment).joinpath(experiment_name)
        path.mkdir(parents=True)
        self.initial_weight_path = str(path.joinpath('initial_weights.h5'))
        self.pkl = path.joinpath('experiment.pkl')
        return path

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('log')
        return state

    def __setstate__(self, state):
        self.__dict__.clear()
        self.__dict__.update(state)

    def load(self):
        with open(self.pkl, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__setstate__(tmp_dict)

    def save(self):
        self.log.info("writing experiment to: {}".format(self.pkl))
        with open(self.pkl, 'wb') as f:
            pickle.dump(self.__getstate__(), f)
