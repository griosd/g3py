import numpy as np
import theano as th
import theano.tensor as tt
from g3py.functions.hypers import Hypers, ones
from g3py.libs.tensors import tt_to_num, debug


class Metric(Hypers):
    def __call__(self, x1, x2):
        return tt.abs_(x1 - x2)

    def gram(self, x1, x2):
        #try:
        return tt_to_num(self(x1[:, self.dims].dimshuffle([0, 'x', 1]), x2[:, self.dims].dimshuffle(['x', 0, 1])))
        #except ValueError:
        #    return tt_to_num(self(x1[:, self.dims].dimshuffle([0, 'x']), x2[:, self.dims].dimshuffle(['x', 0])))

    def input_sensitivity(self):
        return np.ones(self.shape)

    def __str__(self):
        return str(self.__class__.__name__) + '[h=' + str(self.hypers) + ']'
    __repr__ = __str__


class One(Metric):
    def __call__(self, x1, x2):
        return 1


class Delta(Metric):
    def __call__(self, x1, x2):
        return tt.eq((x1 - x2), 0).dot(np.ones(self.shape))

    def gram(self, x1, x2):
        return tt_to_num(self(x1[:, self.dims].dimshuffle([0, 'x', 1]), x2[:, self.dims].dimshuffle(['x', 0, 1])))


class DeltaEq(Metric):
    def __call__(self, x1, x2, eq=0):
        return (tt.eq(x1, eq)*tt.eq(x2, eq)).sum(axis=2)

    def gram(self, x1, x2, eq=0):
        return tt_to_num(self(x1[:, self.dims].dimshuffle([0, 'x', 1]), x2[:, self.dims].dimshuffle(['x', 0, 1]), eq))


class DeltaEq2(Metric):
    def __call__(self, x1, x2, eq1=0, eq2=0):
        return (tt.eq(x1, eq1)*tt.eq(x2, eq2) + tt.eq(x1, eq2)*tt.eq(x2, eq1)).sum(axis=2)

    def gram(self, x1, x2, eq1=0, eq2=0):
        return tt_to_num(self(x1[:, self.dims].dimshuffle([0, 'x', 1]), x2[:, self.dims].dimshuffle(['x', 0, 1]), eq1, eq2))


class Minimum(Metric):
    def __call__(self, x1, x2):
        return tt.prod(tt.minimum(x1-x2*0, x2-x1*0), axis=2)


class Difference(Metric):
    def __call__(self, x1, x2):
        return x1 - x2


class L1(Metric):
    def __call__(self, x1, x2):
        return tt.sum(tt.abs_(x1 - x2))


class L2(Metric):
    def __call__(self, x1, x2):
        return tt.sum(0.5*(x1 - x2)**2)


class ARD(Metric):
    def __init__(self, x, name=None, rate=None):
        super().__init__(x, name)
        self.rate = rate

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.rate is None:
            self.rate = Hypers.FlatExp(parent + 'rate', shape=self.shape)
        self.hypers += [self.rate]

    def input_sensitivity(self):
        return ones(self.shape) * self.rate ** 2


class ARD_L1(ARD):
    def __call__(self, x1, x2):
        return tt.dot(tt.abs_(x1 - x2), self.rate)

    def default_hypers(self, x=None, y=None):
        return {self.rate: 1 / np.abs(x[1:] - x[:-1]).mean(axis=0)}

    def input_sensitivity(self):
        return ones(self.shape) * self.rate


class ARD_L2(ARD):
    def __call__(self, x1, x2):
        return tt.dot((x1 - x2) ** 2, (0.5 * self.rate ** 2))

    def default_hypers(self, x=None, y=None):
        try:
            return {self.rate: 1 / np.abs(x[1:] - x[:-1]).mean(axis=0)}
        except:
            return {}


class ARD_Dot(ARD):
    def __call__(self, x1, x2):
        return tt.dot(x1 * x2, self.rate ** 2)

    def default_hypers(self, x=None, y=None):
        return {self.rate: 1 / ((np.sqrt(np.abs(x)).mean(axis=0)) / np.abs(y).mean(axis=0))}


class ARD_DotBias(ARD):
    def __init__(self, x, name=None, rate=None, bias=None):
        super().__init__(x, name, rate)
        self.bias = bias

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.bias is None:
            self.bias = Hypers.FlatExp(parent + 'bias')
        self.hypers += [self.bias]

    def __call__(self, x1, x2):
        return self.bias + tt.dot(x1 * x2, self.rate ** 2)
        #return self.bias + tt.dot(tt.dot(x1, self.rate), tt.dot(x2, self.rate))

    def default_hypers(self, x=None, y=None):
        return {self.bias: np.abs(y).mean()/np.abs(x).mean(),
                self.rate: np.sqrt(np.abs(y)).mean(axis=0) / np.abs(x).mean(axis=0)}


class PSD(Metric):
    def __init__(self, x, p=1, name=None, rate=None, directions=None):
        super().__init__(x, name)
        self.rate = rate
        self.directions = directions
        self.p = p

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.rate is None:
            self.rate = Hypers.FlatExp(parent + 'rate', shape=self.shape)
        if self.directions is None:
            self.directions = Hypers.FlatExp(parent + 'directions', shape=(self.p, self.shape))
        self.hypers += [self.rate, self.directions]


class PSD_Dot(PSD):
    def __call__(self, x1, x2):
        return tt.dot(tt.dot(x1.T, tt.dot(self.directions.T, self.directions) + tt.diag(self.rate**2)), x2)

    def default_hypers(self, x=None, y=None):
        return {self.rate: 1 / ((np.sqrt(np.abs(x)).mean(axis=0)) / np.abs(y).mean(axis=0)),
                self.directions: np.zeros(self.directions.shape)}

class PSD_L2(PSD):
    def __call__(self, x1, x2):
        d = (x1 - x2)
        M = tt.dot(self.directions.T, self.directions) + tt.diag(self.rate**2)
        d * M
        return tt.dot(M, d)

    def default_hypers(self, x=None, y=None):
        return {self.rate: 1 / ((np.sqrt(np.abs(x)).mean(axis=0)) / np.abs(y).mean(axis=0)),
                self.directions: np.zeros(self.directions.shape)}
