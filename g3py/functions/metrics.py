import numpy as np
import theano as th
import theano.tensor as tt
from g3py.functions.hypers import Hypers
from g3py.libs.tensors import tt_to_num


class Metric(Hypers):
    def __call__(self, x1, x2):
        return tt.abs_(x1 - x2)

    def gram(self, x1, x2):
        return tt_to_num(self(x1.dimshuffle([0, 'x']), x2.dimshuffle(['x', 0])))

    def __str__(self):
        return str(self.__class__.__name__) + '[h=' + str(self.hypers) + ']'
    __repr__ = __str__


class Delta(Metric):
    def __call__(self, x1, x2):
        return x1 == x2


class Minimum(Metric):
    def __call__(self, x1, x2):
        return tt.minimum(x1-x2*0, x2-x1*0)


class L1(Metric):
    def __call__(self, x1, x2):
        return tt.abs_(x1 - x2)


class L2(Metric):
    def __call__(self, x1, x2):
        return ((x1 - x2)**2)/2


class ARD(Metric):
    def __init__(self, x, name=None, scales=None):
        super().__init__(x, name)
        self.scales = scales

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.scales is None:
            self.scales = Hypers.FlatExp(parent+self.name+'_Scales', shape=self.shape)
        self.hypers += [self.scales]


class ARD_L1(ARD):
    def __call__(self, x1, x2):
        return tt.abs_(x1 - x2) / self.scales

    def default_hypers(self, x=None, y=None):
        return {self.scales: np.abs(x[1:] - x[:-1]).mean()}


class ARD_L2(ARD):
    def __call__(self, x1, x2):
        return ((x1 - x2) ** 2) / (2 * self.scales)

    def default_hypers(self, x=None, y=None):
        return {self.scales: np.abs(x[1:] - x[:-1]).mean()}


class ARD_Dot(ARD):
    def __call__(self, x1, x2):
        return tt.dot(x1 * self.scales, x2)

    def default_hypers(self, x=None, y=None):
        return {self.scales: x.mean()}


class ARD_DotBias(ARD):
    def __init__(self, x, name=None, scales=None, bias=None):
        super().__init__(x, name, scales)
        self.bias = bias

    def check_hypers(self, parent=''):
        super().check_hypers()
        if self.bias is None:
            self.bias = Hypers.FlatExp(parent+self.name + '_Bias')
        self.hypers += [self.bias]

    def __call__(self, x1, x2):
        return self.bias + tt.dot(x1 * self.scales, x2)

    def default_hypers(self, x=None, y=None):
        return {self.bias: np.float(0, type=th.config.floatX), self.scales: x.mean()}