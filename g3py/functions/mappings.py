import numpy as np
import theano as th
import theano.tensor as tt
import theano.sandbox.linalg as sT
from g3py.functions.hypers import Hypers


class Mapping(Hypers):
    def __call__(self, x):
        pass

    def inv(self, y):
        pass

    def logdet_dinv(self, y):
        return tt.log(sT.det(tt.jacobian(self.inv(y), y)))


class Identity(Mapping):
    def __init__(self, y=None, name=None):
        super().__init__(y, name)

    def __call__(self, x):
        return x

    def inv(self, y):
        return y

    def logdet_dinv(self, y):
        return 0.0


class LogShifted(Mapping):
    def __init__(self, y, name=None, shift=None):
        super().__init__(y, name)
        self.shift = shift

    def check_hypers(self, parent=''):
        if self.shift is None:
            self.shift = Hypers.Flat(parent+self.name+'_shift', shape=self.shape)
        self.hypers += [self.shift]

    def default_hypers(self, x=None, y=None):
        return {self.shift: y.min() - np.abs(y[1:]-y[:-1]).min()}

    def __call__(self, x):
        return tt.exp(x) + self.hypers

    def inv(self, y):
        return tt.log(y - self.hypers)

    def logdet_dinv(self, y):
        return -tt.sum(tt.log(y - self.hypers))


class BoxCoxShifted(Mapping):
    def __init__(self, y, name='BoxCoxShifted', shift=None, power=None):
        super().__init__(y, name)
        self.shift = shift
        self.power = power

    def check_hypers(self, parent=''):
        if self.shift is None:
            self.shift = Hypers.Flat(parent+self.name+'_shift', shape=self.shape)
        if self.power is None:
            self.power = Hypers.FlatExp(parent+self.name+'_power', shape=self.shape)
        self.hypers += [self.shift, self.power]

    def default_hypers(self, x=None, y=None):
        return {self.shift: np.float32(1.0),
                self.power: np.float32(1.0)}

    def __call__(self, x):
        scaled = self.power*x+1.0
        transformed = tt.sgn(scaled) * tt.abs_(scaled) ** (1.0 / self.power)
        return transformed-self.shift

    def inv(self, y):
        shifted = y+self.shift
        return ((tt.sgn(shifted) * tt.abs_(shifted) ** self.power)-1.0)/self.power

    def logdet_dinv(self, y):
        return (self.power - 1.0)*tt.sum(tt.log(tt.abs_(y+self.shift)))
