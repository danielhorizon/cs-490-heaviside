from setup_paths import *

from glob import iglob
try:
   import cPickle as pickle
except:
   import pickle

def load_experiments(args):
    globpath = str(ROOT_PATH.joinpath('experiments/**').joinpath(args.experiment).joinpath('**/experiment.pkl'))
    experiments = []
    paths = []
    for pkl_path in iglob(globpath):
        paths.append(pkl_path)
        with open(pkl_path, 'rb') as f:
            experiments.append(pickle.loads(f.read()))
    if len(experiments) < 1:
        raise RuntimeError("no paths found in glob: {}".format(globpath))
    return experiments, paths
