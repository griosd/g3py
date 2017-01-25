import numpy as np
import theano as th
import theano.tensor as tt
from g3py.functions.hypers import Hypers
from g3py.functions.metrics import Delta, Minimum, Difference, One, ARD_Dot, ARD_DotBias, ARD_L1, ARD_L2, DeltaEq, DeltaEq2
from g3py.libs.tensors import debug


pi = np.float32(np.pi)


class Kernel(Hypers):
    def __init__(self, x=None, name=None, metric=Delta, var=None):
        if type(metric) is type:
            self.metric = metric(x)
        else:
            self.metric = metric
        super().__init__(x, name)
        self.var = var

    def check_hypers(self, parent=''):
        if self.var is None:
            self.var = Hypers.FlatExp(parent + self.name + '_var')
        if isinstance(self.var, tt.TensorVariable):
            self.hypers += [self.var]
        self.metric.check_hypers(parent + self.name+'_')

    def check_dims(self, x=None):
        super().check_dims(x)
        self.metric.check_dims(x)

    def default_hypers(self, x=None, y=None):
        if isinstance(self.var, tt.TensorVariable):
            if self.metric is None:
                return {self.var: y.var().astype(th.config.floatX)}
            else:
                return {self.var: y.var().astype(th.config.floatX), **self.metric.default_hypers(x, y)}
        else:
            return self.metric.default_hypers(x, y)

    def input_sensitivity(self):
        return self.var*self.metric.input_sensitivity()

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
    def __init__(self, x=None, name=None, metric=ARD_Dot, var=None):
        super().__init__(x, name, metric, var)

    def __call__(self, x1, x2):
        return self.var * self.metric(x1, x2)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.var * self.metric.gram(x1, x1)
        else:
            return self.var * self.metric.gram(x1, x2)


class KernelStationary(Kernel):
    def __init__(self, x=None, name=None, metric=ARD_L2, var=None):
        super().__init__(x, name, metric, var)

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

    def check_dims(self, x=None):
        self.k.check_dims(x)

    def default_hypers_dims(self, x=None, y=None):
        return self.k.default_hypers_dims(x, y)

    @property
    def name(self):
        return str(self.element) + " "+self.op+" " + self.k.name

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

    def check_dims(self, x=None):
        self.k1.check_dims(x)
        self.k2.check_dims(x)

    def default_hypers_dims(self, x=None, y=None):
        return {**self.k1.default_hypers_dims(x, y), **self.k2.default_hypers_dims(x, y)}

    @property
    def name(self):
        return self.k1.name + " "+self.op+" " + self.k2.name

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
        if hasattr(self.k1, 'var') and hasattr(self.k2, 'var'):
            if self.k1.var is None and self.k2.var is None:
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


class KernelMax(KernelComposition):
    def __init__(self, _k1: Kernel, _k2: Kernel):
        super().__init__(_k1, _k2)
        self.op = 'max'

    def __call__(self, x1, x2):
        return tt.maximum(self.k1(x1, x2), self.k2(x1, x2))

    def cov(self, x1, x2=None):
        return tt.maximum(self.k1.cov(x1, x2), self.k2.cov(x1, x2))

    def __str__(self):
        return "max("+str(self.k1)+" , "+str(self.k2)+")"


class KernelEquals(Kernel):
    def __init__(self, x=None, name=None, metric=DeltaEq, eq=0):
        super().__init__(x, name, metric, 1)
        self.eq = eq

    def __call__(self, x1, x2):
        return self.metric(x1, x2, self.eq)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.metric.gram(x1, x1, self.eq)
        else:
            return self.metric.gram(x1, x2, self.eq)


class KernelEquals2(Kernel):
    def __init__(self, x=None, name=None, metric=DeltaEq2, eq1=0, eq2=0):
        super().__init__(x, name, metric, 1)
        self.eq1 = eq1
        self.eq2 = eq2

    def __call__(self, x1, x2):
        return self.metric(x1, x2, self.eq1, self.eq2)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.metric.gram(x1, x1, self.eq1, self.eq2)
        else:
            return self.metric.gram(x1, x2, self.eq1, self.eq2)


class BW(KernelDot):
    def __init__(self, x=None, name=None, metric=Minimum, var=None):
        super().__init__(x, name, metric, var)


class VAR(KernelDot):
    def __init__(self, x=None, name=None, metric=One, var=None):
        super().__init__(x, name, metric, var)

    def __call__(self, x1, x2):
        return self.var

    def cov(self, x1, x2=None):
        return self.var


class LIN(KernelDot):
    def __init__(self, x=None, name=None, metric=ARD_DotBias, var=1):
        super().__init__(x, name, metric, var)


class POL(KernelDot):
    def __init__(self, x=None, p=2, name=None, metric=ARD_DotBias, var=1):
        super().__init__(x, name, metric, var)
        self.p = p

    def __call__(self, x1, x2):
        return self.var * self.metric(x1, x2) ** self.p

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.var * self.metric.gram(x1, x1) ** self.p
        else:
            return self.var * self.metric.gram(x1, x2) ** self.p


class NN(KernelDot):
    def __init__(self, x=None, name=None, metric=ARD_DotBias, var=None):
        super().__init__(x, name, metric, var)

    def __call__(self, x1, x2):
        return self.var * tt.arcsin(2*self.metric(x1, x2)/((1 + 2*self.metric(x1, x1))*(1 + 2*self.metric(x2, x2))))

    def cov(self, x1, x2=None):
        if x2 is None:
            xx = self.metric.gram(x1, x1)
            debug(tt.arcsin(2*xx/((1 + 2*xx)**2)), 'xx')
            return self.var * tt.arcsin(2*xx/((1 + 2*xx)**2))
        else:
            return self.var * tt.arcsin(2*self.metric.gram(x1, x2)/((1 + 2*self.metric.gram(x1, x1))*(1 + 2*self.metric.gram(x2, x2))))


class WN(KernelStationary):
    def __init__(self, x=None, name=None, metric=Delta, var=None):
        super().__init__(x, name, metric, var)

    def __call__(self, x1, x2):
        return self.var * self.metric(x1, x2)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.var * tt.eye(x1.shape[0])
        else:
            return self.var * self.metric.gram(x1, x2)
            #return tt.zeros((x1.shape[0], x2.shape[0]))


class RQ(KernelStationary):
    def __init__(self, x=None, name=None, metric=ARD_L2, var=None, alpha=None):
        super().__init__(x, name, metric, var)
        self.alpha = alpha

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.alpha is None:
            self.alpha = Hypers.FlatExp(parent+self.name+'_alpha')
        self.hypers += [self.alpha]

    def default_hypers(self, x=None, y=None):
        return {self.alpha: np.float32(1.0), **super().default_hypers(x, y)}

    def k(self, d):
        return tt.pow(1 + d / self.alpha, -self.alpha)


class MAT32(KernelStationary):
    def __init__(self, x=None, name=None, metric=ARD_L2, var=None):
        super().__init__(x, name, metric, var)

    def k(self, d):
        d3 = tt.sqrt(3*d)
        return (1 + d3)*tt.exp(-d3)


class MAT52(KernelStationary):
    def __init__(self, x=None, name=None, metric=ARD_L2, var=None):
        super().__init__(x, name, metric, var)

    def k(self, d):
        d5 = tt.sqrt(5*d)
        return (1 + d5 + 5*d/3)*tt.exp(-d5)


class KernelStationaryExponential(KernelStationary):
    def k(self, d):
        return tt.exp(-d)


class OU(KernelStationaryExponential):
    def __init__(self, x=None, name=None, metric=ARD_L1, var=None):
        super().__init__(x, name, metric, var)


class SE(KernelStationaryExponential):
    def __init__(self, x=None, name=None, metric=ARD_L2, var=None):
        super().__init__(x, name, metric, var)


class KernelPeriodic(KernelStationary):
    def __init__(self, x=None, name=None, metric=Difference, var=None, periods=None, rate=None):
        super().__init__(x, name, metric, var)
        self.periods = periods
        self.rate = rate

    def check_hypers(self, parent=''):
        super().check_hypers(parent=parent)
        if self.periods is None:
            self.periods = Hypers.FlatExp(parent + self.name + '_per', shape=self.shape)
        if self.rate is None:
            self.rate = Hypers.FlatExp(parent + self.name + '_rate', shape=self.shape)
        if isinstance(self.rate, tt.TensorVariable):
            self.hypers += [self.rate]
        if isinstance(self.periods, tt.TensorVariable):
            self.hypers += [self.periods]

    def default_hypers(self, x=None, y=None):
        return {self.periods: (x.max(axis=0)-x.min(axis=0)),
                self.rate: 1 / np.abs(x[1:] - x[:-1]).mean(axis=0),
                **super().default_hypers(x, y)}


class COS(KernelPeriodic):
    def __init__(self, x=None, name=None, metric=Difference, var=None, periods=None):
        super().__init__(x, name, metric, var)
        self.periods = periods
        self.rate = None

    def k(self, d):
        return tt.prod(tt.cos(2 * pi * d/self.periods), axis=2)


class PER(KernelPeriodic):
    def k(self, d):
        return tt.exp(2 * tt.dot(tt.sin(pi * d/self.periods) ** 2, self.rate))


class SM(KernelPeriodic):
    def k(self, d):
        return tt.exp(-2 * pi ** 2 * tt.dot(d ** 2, self.rate ** 2)) * tt.prod(tt.cos(2 * pi * d / self.periods), axis=2)


