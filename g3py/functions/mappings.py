import numpy as np
import theano.tensor as tt
import theano.sandbox.linalg as sT
from g3py.functions.hypers import Hypers
from g3py.libs.tensors import tt_to_num, debug

class Mapping(Hypers):
    def __call__(self, x):
        pass

    def inv(self, y):
        pass

    def logdet_dinv(self, y):
        pass

    def automatic_logdet_dinv(self, y):
        return debug(tt.log(sT.det(debug(tt.jacobian(self.inv(y), y), 'jacobian_inv'))), 'automatic_logdet_dinv')


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


class Logistic(Mapping):
    def __init__(self, y, name='Logistic', lower=None, high=None, location=None, scale=None):
        super().__init__(y, name)
        self.lower = lower
        self.high = high
        self.location = location
        self.scale = scale

    def check_hypers(self, parent=''):
        if self.lower is None:
            self.lower = Hypers.Flat(parent+self.name+'_lower', shape=self.shape)
        if self.high is None:
            self.high = Hypers.FlatExp(parent+self.name+'_high', shape=self.shape)
        if self.location is None:
            self.location = Hypers.Flat(parent+self.name+'_location', shape=self.shape)
        if self.scale is None:
            self.scale = Hypers.FlatExp(parent+self.name+'_scale', shape=self.shape)
        self.hypers += [self.lower, self.high, self.location, self.scale]

    def default_hypers(self, x=None, y=None):
        return {self.lower: np.float32(1.5*np.min(y) - 0.5*np.max(y)),
                self.high: np.float32(2.0*(np.max(y) - np.min(y))),
                self.location: np.float32(np.mean(y)),
                self.scale: np.float32(np.std(y))}

    def __call__(self, x):
        return self.lower + self.high*(0.5 + 0.5*tt.tanh((x - self.location)/(2*self.scale)))

    def inv(self, y):
        p = tt.switch(y < self.lower, 0, tt.switch(y > self.lower + self.high, 1, (y - self.lower) / self.high))
        return debug(self.location + self.scale*tt_to_num(tt.log(p / (1 - p))), 'inv')

    def logdet_dinv(self, y):
        p = tt.switch(y < self.lower, 0, tt.switch(y > self.lower + self.high, 1, (y - self.lower) / self.high))
        return debug(tt.sum(tt_to_num(tt.log(self.scale / (self.high * p * (1-p))))), 'logdet_dinv')