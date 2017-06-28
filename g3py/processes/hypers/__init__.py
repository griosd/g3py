#from .means import *
#from .metrics import *
#from .kernels import *
#from .mappings import *
import numpy as np
import theano as th
import theano.tensor as tt
import pymc3 as pm
from ...libs import DictObj


class Hypers:
    def __init__(self, x=None, name=None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.hypers = []
        self.shape = None
        self.dims = None
        self.potential = None
        self.check_dims(x)

    def __str__(self):
        if len(self.hypers) is 0:
            return str(self.__class__.__name__)
        else:
            return str(self.__class__.__name__)+'[h='+str(self.hypers) + ']'
    __repr__ = __str__

    def check_dims(self, x=None):
        if self.shape is not None:
            return
        if x is not None:
            if type(x) is tuple:
                domain, self.dims = x
                if len(domain.shape) > 1:
                    self.shape = domain.shape[1]
                else:
                    self.shape = 1
            else:
                if len(x.shape) > 1:
                    self.shape = x.shape[1]
                else:
                    self.shape = 1
                self.dims = slice(0, self.shape)
        else:
            self.shape = None
            self.dims = slice(None)

    def check_hypers(self, parent=''):
        pass

    def default_hypers(self, x=None, y=None):
        return {}

    def default_hypers_dims(self, x=None, y=None):
        return get_hypers_floatX(self.default_hypers(x[:, self.dims], y))

    def set_potential(self, hypers='', reg='L1', c=1):
        self.potential = (hypers, reg, c)

    def check_potential(self):
        if self.potential is None:
            return
        hypers, reg, c = self.potential
        if reg == 'L1':
            pot = -tt.sum([tt.abs_(k) for k in self.hypers if k.name.find(hypers) > 0])
        elif reg == 'L2':
            pot = -tt.sum(([k**2 for k in self.hypers if k.name.find(hypers) > 0]))
        else:
            pot = 0
        return pm.Potential(self.name+'_'+hypers+'_'+reg, c * pot)


class Freedom(Hypers):
    def __init__(self, x=None, name=None, degree=None, bound=2):
        super().__init__(x, name)
        self.degree = degree
        self.bound = bound

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.degree is None:
            self.degree = Hypers.FlatExp(parent+self.name+'_degree')
        self.hypers += [self.degree]

    def default_hypers(self, x=None, y=None):
        return {self.degree: y.shape[0].astype(th.config.floatX)}

    def __call__(self, x=None):
        return self.bound + self.degree


def get_hypers_floatX(params):
    paramsX = DictObj()
    for k, v in params.items():
        paramsX[k] = np.float32(v)
    return paramsX


def zeros(shape):
    return np.zeros(shape, dtype=th.config.floatX)


def ones(shape):
    return np.ones(shape, dtype=th.config.floatX)

def cvalues(shape, val):
    return np.float32(np.ones(shape, dtype=th.config.floatX) *val)


class PositiveFlat(pm.Continuous):
    """
    Uninformative log-likelihood that returns 0 regardless of
    the passed value.
    """

    def __init__(self, *args, **kwargs):
        super(PositiveFlat, self).__init__(*args, **kwargs)
        self.median = 1

    def random(self, point=None, size=None, repeat=None):
        raise ValueError('Cannot sample from Flat distribution')

    def logp(self, value):
        return tt.switch(value > 0, 0, -np.inf)


class LogIdTransform(pm.distributions.transforms.ElemwiseTransform):
    name = "logid"

    def backward(self, x):
        return tt.switch(x < 0, tt.exp(x), x + 1.0)

    def forward(self, x):
        return tt.switch(x < 1, tt.log(x), x - 1.0)


class NonTransformLog(pm.distributions.transforms.ElemwiseTransform):
    name = "log"

    def backward(self, x):
        return tt.exp(x)

    def forward(self, x):
        return tt.log(x)

    def jacobian_det(self, x):
        return tt.switch(tt.exp(x) > 1e-6, 0, -np.inf)

non_transform_log = NonTransformLog()
