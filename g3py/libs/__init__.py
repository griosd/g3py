from .data import *
from .plots import *
from .tensors import *
from .traces import *
from .experiments import *
from theano.ifelse import ifelse
import warnings

def clone(c):
    return copy(c)


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


def nan_to_high(x):
    return np.where(np.isfinite(x), x, 1.0e100)


class MaxTime(object):
    def __init__(self, max_sec=60):
        self.max_sec = max_sec + time.time()

    def __call__(self, xk=None):

        if time.time() > self.max_sec:
            raise Exception("Terminating: time limit reached")
