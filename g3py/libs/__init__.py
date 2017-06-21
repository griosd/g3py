import os
import _pickle as pickle
import time
from copy import copy
from .data import *
from .plots import *
#from .tensors import *
#from .traces import *
#from .experiments import *
#from theano.ifelse import ifelse
#import warnings


class DictObj(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class MaxTime(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec + time.time()

    def __call__(self, xk=None):

        if time.time() > self.max_sec:
            raise Exception("Terminating: time limit reached")


def clone(c):
    return copy(c)


def nan_to_high(x):
    return np.where(np.isfinite(x), x, 1.0e100)


def save_pkl(to_pkl, path='file.pkl'):
    os.makedirs(path[:path.rfind('/')], exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(to_pkl, f, protocol=-1)


def load_pkl(path='file.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)