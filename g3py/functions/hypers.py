import numpy as np
import theano as th
import theano.tensor as tt
import pymc3 as pm


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


def trans_hypers(hypers):
    trans = {}
    for k, v in hypers.items():
        if type(k) is pm.model.TransformedRV:
            trans[k.transformed] = k.transformed.distribution.transform_used.forward(v).eval()
        else:
            trans[k] = v
    return trans


class Hypers:
    def __init__(self, x=None, name=None):
        if x is not None:
            if type(x) is tuple:
                domain, self.dims = x
                self.shape = domain.shape[1]
            else:
                self.shape = x.shape[1]
                self.dims = slice(0, self.shape)
            #if self.shape == 1:
            #    self.shape = ()
        else:
            self.shape = ()
            self.dims = slice(None)
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.hypers = []

    def __str__(self):
        if len(self.hypers) is 0:
            return str(self.__class__.__name__)
        else:
            return str(self.__class__.__name__)+'[h='+str(self.hypers) + ']'
    __repr__ = __str__

    def check_hypers(self, parent=''):
        pass

    def default_hypers(self, x=None, y=None):
        return {}

    def default_hypers_dims(self, x=None, y=None):
        return self.default_hypers(x[:, self.dims], y)

    @staticmethod
    def Flat(name, shape=(), testval=zeros):
        with modelcontext():
            return pm.Flat(name, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def ExpFlat(name, shape=(), testval=zeros):
        with modelcontext():
            return tt.exp(pm.Flat(name, shape=shape, testval=testval(shape), dtype=th.config.floatX))
    @staticmethod
    def FlatExp(name, shape=(), testval=ones):
        with modelcontext():
            return pm.Flat(name, transform=pm.distributions.transforms.log, shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def FlatExpId(name, shape=(), testval=ones):
        with modelcontext():
            return pm.Flat(name, transform=LogIdTransform(), shape=shape, testval=testval(shape), dtype=th.config.floatX)
    @staticmethod
    def Exponential(name, lam=ones, shape=(), testval=ones):
        with modelcontext():
            return pm.Exponential(name, shape=shape, lam=lam(shape), testval=testval(shape), dtype=th.config.floatX)
