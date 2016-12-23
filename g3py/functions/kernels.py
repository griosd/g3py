import numpy as np
import theano as th
#import pymc3 as pm
import theano.tensor as tt
from g3py.functions.hypers import Hypers

from g3py.functions.metrics import Delta, Minimum, L1, ARD_Dot, ARD_DotBias, ARD_L1, ARD_L2

pi = np.float32(np.pi)


class Kernel(Hypers):
    def __init__(self, x, metric, name=None, var=None):
        super().__init__(x, name)
        self.metric = metric(x)
        self.var = var

    def check_hypers(self, parent=''):
        if self.var is None:
            self.var = Hypers.FlatExp(parent + self.name + '_Var')
        if isinstance(self.var, tt.TensorVariable):
            self.hypers += [self.var]
        self.metric.check_hypers(parent+self.name+'_')

    def default_hypers(self, x=None, y=None):
        if isinstance(self.var, tt.TensorVariable):
            return {self.var: y.var().astype(th.config.floatX), **self.metric.default_hypers(x, y)}
        else:
            return self.metric.default_hypers(x, y)

    def __call__(self, x1, x2):
        pass

    def cov(self, x1, x2=None):
        pass

    def __mul__(self, other):
        if issubclass(type(other), Kernel):
            return KernelProd(self, other)
        else:
            return KernelScale(self, other)
    __imul__ = __mul__

    def __rmul__(self, other):
        if issubclass(type(other), Kernel):
            return KernelProd(other, self)
        else:
            return KernelScale(self, other)

    def __add__(self, other):
        if issubclass(type(other), Kernel):
            return KernelSum(self, other)
        else:
            return KernelShift(self, other)
    __iadd__ = __add__

    def __radd__(self, other):
        if issubclass(type(other), Kernel):
            return KernelSum(other, self)
        else:
            return KernelShift(self, other)

    def __str__(self):
        return str(self.__class__.__name__)+'[m='+str(self.metric)+',h='+str(self.hypers) + ']'
    __repr__ = __str__


class KernelDot(Kernel):
    def __init__(self, x, metric=ARD_Dot, name='KernelDot', var=None):
        super().__init__(x, metric, name, var)

    def __call__(self, x1, x2):
        return self.var * self.metric(x1, x2)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.var * self.metric.gram(x1, x1)
        else:
            return self.var * self.metric.gram(x1, x2)


class KernelStationary(Kernel):
    def __init__(self, x, metric=ARD_L2, name='KernelStationary', var=None):
        super().__init__(x, metric, name, var)

    def k(self, d):
        return d

    def __call__(self, x1, x2):
        return self.var * self.k(self.metric(x1, x2))

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.var * self.k(self.metric.gram(x1, x1))
        else:
            return self.var * self.k(self.metric.gram(x1, x2))


class KernelOperation(Kernel):
    def __init__(self, _k: Kernel, _element):
        self.k = _k
        self.element = _element
        self.hypers = []
        self.op = 'op'

    def check_hypers(self, parent=''):
        self.k.check_hypers(parent=parent)
        self.hypers = self.k.hypers

    def default_hypers(self, x=None, y=None):
        return self.k.default_hypers(x, y)

    def __str__(self):
        return str(self.element) + " "+self.op+" " + str(self.k)
    __repr__ = __str__


class KernelComposition(Kernel):
    def __init__(self, _k1: Kernel, _k2: Kernel):
        self.k1 = _k1
        self.k2 = _k2
        self.op = 'op'

    def check_hypers(self, parent=''):
        self.k1.check_hypers(parent=parent)
        self.k2.check_hypers(parent=parent)
        self.hypers = self.k1.hypers + self.k2.hypers

    def default_hypers(self, x=None, y=None):
        return {**self.k1.default_hypers(x, y), **self.k2.default_hypers(x, y)}

    def __str__(self):
        return str(self.k1) + " "+self.op+" " + str(self.k2)
    __repr__ = __str__


class KernelScale(KernelOperation):
    def __call__(self, x1, x2):
        return self.element * self.k(x1, x2)

    def cov(self, x1, x2=None):
        return self.element * self.k.cov(x1, x2)

    def __str__(self):
        return str(self.element) + " * " + str(self.k)


class KernelShift(KernelOperation):
    def __call__(self, x1, x2):
        return self.element + self.k(x1, x2)

    def cov(self, x1, x2=None):
        return self.element + self.k.cov(x1, x2)

    def __str__(self):
        return str(self.element) + " + " + str(self.k)


class KernelProd(KernelComposition):
    def __init__(self, _k1: Kernel, _k2: Kernel):
        super().__init__(_k1, _k2)
        if self.k1.var is None:
            self.k1.var = 1.0
        elif self.k2.var is None:
            self.k2.var = 1.0
        self.op = '*'

    def __call__(self, x1, x2):
        return self.k1(x1, x2) * self.k2(x1, x2)

    def cov(self, x1, x2=None):
        return self.k1.cov(x1, x2) * self.k2.cov(x1, x2)

    def __str__(self):
        return str(self.k1) + " * " + str(self.k2)


class KernelSum(KernelComposition):
    def __init__(self, _k1: Kernel, _k2: Kernel):
        super().__init__(_k1, _k2)
        self.op = '+'

    def __call__(self, x1, x2):
        return self.k1(x1, x2) + self.k2(x1, x2)

    def cov(self, x1, x2=None):
        return self.k1.cov(x1, x2) + self.k2.cov(x1, x2)

    def __str__(self):
        return str(self.k1) + " + " + str(self.k2)


class BW(KernelDot):
    def __init__(self, x=None, metric=Minimum, name=None, var=None):
        super().__init__(x, metric, name, var)


class LIN(KernelDot):
    def __init__(self, x, metric=ARD_DotBias, name=None, var=None):
        super().__init__(x, metric, name, 1)


class POL(KernelDot):
    def __init__(self, x, p=2, metric=ARD_DotBias, name=None, var=None):
        super().__init__(x, metric, name, var)
        self.p = p

    def __call__(self, x1, x2):
        return self.var * self.metric(x1, x2) ** self.p

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.var * self.metric.gram(x1, x1) ** self.p
        else:
            return self.var * self.metric.gram(x1, x2) ** self.p


class NN(KernelDot):
    def __init__(self, x, p=2, metric=ARD_DotBias, name=None, var=None):
        super().__init__(x, metric, name, var)
        self.p = p

    def __call__(self, x1, x2):
        return self.var * tt.arcsin(2*self.metric(x1, x2)/((1 + 2*self.metric(x1, x1))*(1 + 2*self.metric(x2, x2))))

    def cov(self, x1, x2=None):
        if x2 is None:
            xx = self.metric.gram(x1, x1)
            return self.var * tt.arcsin(2*xx/((1 + 2*xx)**2))
        else:
            return self.var * tt.arcsin(2*self.metric.gram(x1, x2)/((1 + 2*self.metric.gram(x1, x1))*(1 + 2*self.metric.gram(x2, x2))))




class WN(KernelStationary):
    def __init__(self, x=None, metric=Delta, name=None, var=None):
        super().__init__(x, metric, name, var)

    def __call__(self, x1, x2):
        return self.var * self.metric(x1, x2)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.var * tt.eye(x1.shape[0])
        else:
            return 0.0


class RQ(KernelStationary):
    def __init__(self, x, metric=ARD_L2, name=None, var=None, alpha=None):
        super().__init__(x, metric, name, var)
        self.alpha = alpha

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.alpha is None:
            self.alpha = Hypers.FlatExp(parent+self.name+'_ALPHA')
        self.hypers += [self.alpha]

    def default_hypers(self, x=None, y=None):
        return {self.alpha: np.float32(1.0), **super().default_hypers(x, y)}

    def k(self, d):
        return tt.pow(1 + d / self.alpha, -self.alpha)


class COS(KernelStationary):
    def __init__(self, x, metric=ARD_L1, name=None, var=None, alpha=None):
        super().__init__(x, metric, name, var)

    def k(self, d):
        return tt.cos(d)


class MAT32(KernelStationary):
    def __init__(self, x, metric=ARD_L1, name=None, var=None, alpha=None):
        super().__init__(x, metric, name, var)

    def k(self, d):
        d3 = np.sqrt(3)*d
        return (1 + d3)*tt.exp(-d3)


class MAT52(KernelStationary):
    def __init__(self, x, metric=ARD_L2, name=None, var=None, alpha=None):
        super().__init__(x, metric, name, var)

    def k(self, d):
        d5 = np.sqrt(5)*d
        return (1 + d5 + (d5**2)/3)*tt.exp(-d5)


class KernelStationaryExponential(KernelStationary):
    def k(self, d):
        return tt.exp(-d)


class OU(KernelStationaryExponential):
    def __init__(self, x, metric=ARD_L1, name=None, var=None):
        super().__init__(x, metric, name, var)


class SE(KernelStationaryExponential):
    def __init__(self, x, metric=ARD_L2, name=None, var=None):
        super().__init__(x, metric, name, var)


class PER(KernelStationary):
    def __init__(self, x, metric=L1, name=None, var=None, p=None, l=None):
        super().__init__(x, metric, name, var)
        self.p = p
        self.l = l

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.p is None:
            self.p = Hypers.FlatExp(parent+self.name+'_P')
        if self.l is None:
            self.l = Hypers.FlatExp(parent+self.name+'_L')
        self.hypers += [self.p, self.l]

    def default_hypers(self, x=None, y=None):
        return {self.p: np.float32(1.0),
                self.l: np.float32(1.0), **super().default_hypers(x, y)}

    def k(self, d):
        return tt.exp(2*(tt.sin(pi*d / self.p)**2) / self.l)


class SM(KernelStationary):
    def __init__(self, x, metric=L1, name=None, var=None, m=None, s=None):
        super().__init__(x, metric, name, var)
        self.m = m
        self.s = s

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.m is None:
            self.m = Hypers.Flat(parent+self.name+'_M')
        if self.s is None:
            self.s = Hypers.FlatExp(parent+self.name+'_S')
        self.hypers += [self.m, self.s]

    def default_hypers(self, x=None, y=None):
        return {self.m: np.float32(1.0),
                self.s: np.float32(1.0), **super().default_hypers(x, y)}

    def k(self, d):
        return tt.exp(-2 * pi**2 * d**2 * self.s) * tt.cos(2 * pi * d * self.m)


