import numpy as np
import theano as th
import theano.tensor as tt
from g3py.functions.hypers import Hypers, ones
from g3py.libs.tensors import tt_to_num


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


class Delta(Metric):
    def __call__(self, x1, x2):
        return x1 == x2


class Minimum(Metric):
    def __call__(self, x1, x2):
        return tt.prod(tt.minimum(x1-x2*0, x2-x1*0))


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
    def __init__(self, x, name=None, scales=None):
        super().__init__(x, name)
        self.scales = scales

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.scales is None:
            self.scales = Hypers.FlatExp(parent+self.name+'_Scales', shape=self.shape)
        self.hypers += [self.scales]

    def input_sensitivity(self):
        return ones(self.shape) / self.scales**2


class ARD_L1(ARD):
    def __call__(self, x1, x2):
        return tt.dot(tt.abs_(x1 - x2), 1 / self.scales)

    def default_hypers(self, x=None, y=None):
        return {self.scales: np.abs(x[1:]-x[:-1]).mean(axis=0)}

    def input_sensitivity(self):
        return ones(self.shape) / self.scales


class ARD_L2(ARD):
    def __call__(self, x1, x2):
        return tt.dot((x1 - x2) ** 2, 1 / (2 * self.scales**2))

    def default_hypers(self, x=None, y=None):
        return {self.scales: np.abs(x[1:]-x[:-1]).mean(axis=0)}


class ARD_Dot(ARD):
    def __call__(self, x1, x2):
        return tt.dot(tt.dot(x1, 1/self.scales), tt.dot(x2, 1/self.scales))

    def default_hypers(self, x=None, y=None):
        return {self.scales: (np.sqrt(np.abs(y)).mean(axis=0))/np.abs(x).mean(axis=0)}


class ARD_DotBias(ARD):
    def __init__(self, x, name=None, scales=None, bias=None):
        super().__init__(x, name, scales)
        self.bias = bias

    def check_hypers(self, parent=''):
        super().check_hypers()
        if self.bias is None:
            self.bias = Hypers.FlatExp(parent+self.name + '_Bias', shape=1)
        self.hypers += [self.bias]

    def __call__(self, x1, x2):
        return self.bias + tt.dot(tt.dot(x1, 1/self.scales), tt.dot(x2, 1/self.scales))

    def default_hypers(self, x=None, y=None):
        return {self.bias: (np.abs(y).mean(axis=0))/np.abs(x).mean(axis=0),
                self.scales: (np.sqrt(np.abs(y)).mean(axis=0))/np.abs(x).mean(axis=0)}