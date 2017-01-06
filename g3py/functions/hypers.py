import numpy as np
import pandas as pd
import theano as th
import theano.tensor as tt
import pymc3 as pm
from g3py.libs import DictObj


class LogIdTransform(pm.distributions.transforms.ElemwiseTransform):
    name = "logid"
    def backward(self, x):
        return tt.switch(x<0,tt.exp(x),x+1.0)
    def forward(self, x):
        return tt.switch(x<1,tt.log(x),x-1.0)


def modelcontext(model=None):
    return pm.modelcontext(model)


def zeros(shape):
    return np.zeros(shape, dtype=th.config.floatX)


def ones(shape):
    return np.ones(shape, dtype=th.config.floatX)


def cvalues(shape, val):
    return np.float32(np.ones(shape, dtype=th.config.floatX) *val)


def trans_hypers(hypers):
    trans = DictObj()
    for k, v in hypers.items():
        if type(k) is pm.model.TransformedRV:
            trans[k.transformed] = k.transformed.distribution.transform_used.forward(v).eval()
        else:
            trans[k] = v
    return trans


def def_space(space=None, name=None, squeeze=False):
    if space is None:
        space = np.arange(0, 2, dtype=th.config.floatX)
        space = space[:, None]
    elif np.isscalar(space):
        space_x = np.arange(0, space, dtype=th.config.floatX)
        space_x = space_x[None, :]
        return th.shared(space_x, name, borrow=True), None, None

    if squeeze:
        space = np.squeeze(space)
        if type(space) is np.ndarray:
            space_x = space.astype(th.config.floatX)
            if name is None:
                space_th = None
            else:
                space_th = th.shared(space_x, name, borrow=True)
            if len(space_x.shape) == 1 or space_x.shape[1] == 1:
                space_t = np.squeeze(space_x)
            else:
                space_t = np.arange(len(space_x), dtype=th.config.floatX)
        else:
            space_t = space.index
            space_x = space.astype(th.config.floatX)
            if name is None:
                space_th = None
            else:
                space_th = th.shared(space_x.values, name, borrow=True)
        return space_th, space_x, space_t

    if type(space) is np.ndarray:
        if len(space.shape) < 2:
            space = space[:, None]
        space_x = space.astype(th.config.floatX)
        if name is None:
            space_th = None
        else:
            space_th = th.shared(space_x, name, borrow=True)
        if len(space_x.shape) == 1 or space_x.shape[1] == 1:
            space_t = np.squeeze(space_x)
        else:
            space_t = np.arange(len(space_x), dtype=th.config.floatX)
    else:
        space_t = space.index
        space_x = space.astype(th.config.floatX)
        if len(space.shape) < 2:
            space_x = pd.DataFrame(space_x)
        if name is None:
            space_th = None
        else:
            space_th = th.shared(space_x.values, name, borrow=True)
    return space_th, space_x, space_t


class Hypers:
    def __init__(self, x=None, name=None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.hypers = []
        self.shape = None
        self.dims = None
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


    @staticmethod
    def Null(name, shape=(), testval=zeros):
        with modelcontext():
            return pm.NoDistribution(name, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def Flat(name, shape=(), testval=zeros):
        with modelcontext():
            return pm.Flat(name, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def ExpFlat(name, shape=(), testval=ones):
        with modelcontext():
            return pm.Flat(name, transform=pm.distributions.transforms.log, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def FlatExp(name, shape=(), testval=ones):
        with modelcontext():
            return pm.Flat(name, transform=non_transform_log, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def FlatExpId(name, shape=(), testval=ones):
        with modelcontext():
            return pm.Flat(name, transform=LogIdTransform(), shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def Exponential(name, lam=ones, shape=(), testval=ones):
        with modelcontext():
            return pm.Exponential(name, shape=shape, lam=lam(shape), testval=testval(shape), dtype=th.config.floatX)


class NonTransformLog(pm.distributions.transforms.ElemwiseTransform):
    name = "log"
    def backward(self, x):
        return tt.exp(x)

    def forward(self, x):
        return tt.log(x)

    def jacobian_det(self, x):
        return 0
non_transform_log = NonTransformLog()


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