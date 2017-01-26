import numpy as np
import theano.tensor as tt
import theano.sandbox.linalg as sT
from g3py.functions.hypers import Hypers, ones, zeros
from g3py.libs.tensors import tt_to_num, debug, inverse_function


class Mapping(Hypers):

    def __call__(self, z):
        return inverse_function(self.inv, z)

    def inv(self, y):
        pass

    def logdet_dinv(self, y):
        # return debug(tt.log(sT.det(debug(tt.jacobian(self.inv(y), y), 'jacobian_inv'))), 'automatic_logdet_dinv')
        return tt.sum(debug(tt.log(tt.diag(debug(tt.jacobian(self.inv(y), y), 'jacobian_inv'))), 'automatic_logdet_dinv'))

    def __mul__(self, other):
        if issubclass(type(other), Mapping):
            return MappingInvProd(self, other)
        else:
            return MappingInvProd(self, other)
    __imul__ = __mul__
    __rmul__ = __mul__

    def __matmul__(self, other):
        if issubclass(type(other), Mapping):
            return MappingComposed(self, other)
        else:
            return MappingComposed(self, other)
    __imatmul__ = __matmul__
    __rmatmul__ = __matmul__


class MappingOperation(Mapping):
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2
        self.op = 'op'

    def check_hypers(self, parent=''):
        self.m1.check_hypers(parent=parent)
        self.m2.check_hypers(parent=parent)
        self.hypers = self.m1.hypers + self.m2.hypers

    def check_dims(self, x=None):
        self.m1.check_dims(x)
        self.m2.check_dims(x)

    def default_hypers_dims(self, x=None, y=None):
        return {**self.m1.default_hypers_dims(x, y), **self.m2.default_hypers_dims(x, y)}

    def __str__(self):
        return str(self.m1) + " "+self.op+" " + str(self.m2)
    __repr__ = __str__


class MappingComposed(MappingOperation):
    def __init__(self, m1: Mapping, m2: Mapping):
        super().__init__(m1, m2)
        self.op = '@'

    def __call__(self, x):
        return self.m1(self.m2(x))

    def inv(self, y):
        return self.m2.inv(self.m1.inv(y))

    def logdet_dinv(self, y):
        return self.m2.logdet_dinv(self.m1.inv(y)) + self.m1.logdet_dinv(y)


class MappingInvProd(MappingOperation):
    def __init__(self, m1: Mapping, m2: Mapping):
        super().__init__(m1, m2)
        self.op = '*'

    def __call__(self, x):
        pass

    def inv(self, y):
        return self.m1.inv(y) * self.m2.inv(y)

    def logdet_dinv(self, y):
        return self.m2.logdet_dinv(self.m1.inv(y)) + self.m1.logdet_dinv(y)


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
    def __init__(self, y=None, name=None, shift=None):
        super().__init__(y, name)
        self.shift = shift

    def check_hypers(self, parent=''):
        if self.shift is None:
            self.shift = Hypers.Flat(parent+self.name+'_shift')
        self.hypers += [self.shift]

    def default_hypers(self, x=None, y=None):
        return {self.shift: np.array(y.min() - np.abs(y[1:]-y[:-1]).min())}

    def __call__(self, x):
        return tt.exp(x) + self.shift

    def inv(self, y):
        return tt.log(y - self.shift)

    def logdet_dinv(self, y):
        return -tt.sum(tt.log(y - self.shift))


class BoxCoxShifted(Mapping):
    def __init__(self, y=None, name=None, shift=None, scale=None, power=None):
        super().__init__(y, name)
        self.shift = shift
        self.scale = scale
        self.power = power

    def check_hypers(self, parent=''):
        if self.shift is None:
            self.shift = Hypers.Flat(parent+self.name+'_shift')
        if self.scale is None:
            self.scale = Hypers.FlatExp(parent+self.name+'_scale')
        if self.power is None:
            self.power = Hypers.FlatExp(parent+self.name+'_power')
        self.hypers += [self.shift, self.scale, self.power]

    def default_hypers(self, x=None, y=None):
        return {self.shift: np.float32(1.0),#np.array(y.min() - np.abs(y[1:]-y[:-1]).min()),
                self.scale: np.float32(1.0),
                self.power: np.float32(1.0)}

    def __call__(self, x):
        scaled = self.power*x+1.0
        transformed = tt.sgn(scaled) * tt.abs_(scaled) ** (1.0 / self.power)
        return transformed/self.scale-self.shift

    def inv(self, y):
        shifted = self.scale*(y+self.shift)
        return ((tt.sgn(shifted) * tt.abs_(shifted) ** self.power)-1.0)/self.power

    def logdet_dinv(self, y):
        return (self.power - 1.0)*tt.sum(tt.log(tt.abs_(self.scale*(y+self.shift)))) + y.shape[0]*tt.log(self.scale)


class SinhMapping(Mapping):
    def __init__(self, y=None, name=None, shift=None, scale=None):
        super().__init__(y, name)
        self.shift = shift
        self.scale = scale

    def check_hypers(self, parent=''):
        if self.shift is None:
            self.shift = Hypers.Flat(parent+self.name+'_shift')
        if self.scale is None:
            self.scale = Hypers.FlatExp(parent+self.name+'_scale')
        self.hypers += [self.shift, self.scale]

    def default_hypers(self, x=None, y=None):
        return {self.shift: np.float(0.0),#np.float32(1.0),#
                self.scale: np.abs(y).max()}

    def __call__(self, x):
        return tt.sinh(self.scale*x)/self.scale - self.shift

    def inv(self, y):
        return tt.arcsinh(self.scale*(y+self.shift))/self.scale

    def logdet_dinv(self, y):
        return -0.5*tt.sum(tt.log1p((self.scale*(y+self.shift))**2 ))



class Logistic(Mapping):
    def __init__(self, y=None, name=None, lower=None, high=None, location=None, scale=None):
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


class WarpingTanh(Mapping):
    def __init__(self, y=None, n=1, name=None, a=None, b=None, c=None):
        super().__init__(y, name)
        self.n = n
        self.a = a
        self.b = b
        self.c = c

    def check_hypers(self, parent=''):
        if self.a is None:
            self.a = Hypers.FlatExp(parent+self.name+'_a', shape=self.n)
        if self.b is None:
            self.b = Hypers.FlatExp(parent+self.name+'_b', shape=self.n)
        if self.c is None:
            self.c = Hypers.Flat(parent+self.name+'_c', shape=self.n)
        self.hypers += [self.a, self.b, self.c]

    def default_hypers(self, x=None, y=None):
        return {self.a: 0.1 * ones(self.n)*np.abs(y).max() / self.n,
                self.b: 0.1 * ones(self.n)/np.abs(y).max(),
                self.c: ones(self.n)*np.mean(y)}

    def inv(self, y):
        z = y.dimshuffle(0, 'x')
        return y + tt.dot(tt.tanh(self.b*(z + self.c)), self.a).reshape(y.shape)


class WarpingBoxCox(Mapping):
    def __init__(self, y=None, n=1, name=None, shift=None, power=None, w=None):
        super().__init__(y, name)
        self.n = n
        self.shift = shift
        self.power = power
        self.w = w

    def check_hypers(self, parent=''):
        if self.shift is None:
            self.shift = Hypers.FlatExp(parent+self.name+'_shift', shape=self.n)
        if self.power is None:
            self.power = Hypers.FlatExp(parent+self.name+'_power', shape=self.n)
        if self.w is None:
            self.w = Hypers.FlatExp(parent+self.name+'_w', shape=self.n)
        self.hypers += [self.shift, self.power]

    def default_hypers(self, x=None, y=None):
        return {self.w: 0.1 * ones(self.n)*np.abs(y).max()/self.n,
                self.shift: ones(self.n)*np.array(y.min() - np.abs(y[1:]-y[:-1]).min()/self.n),#np.float32(1.0),#
                self.power: ones(self.n)*np.float32(1.0)}

    def inv(self, y):
        z = y.dimshuffle(0, 'x')
        shifted = z+self.shift
        return tt.dot(((tt.sgn(shifted) * tt.abs_(shifted) ** self.power)-1.0)/self.power, self.w).reshape(y.shape)