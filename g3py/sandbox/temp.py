import theano.tensor as tt
from .hypers import Hypers
from .metrics import Delta, ARD_Dot, ARD_DotBias, ARD_L1, ARD_L2


class Kernel:
    def __init__(self, x, _metric, name='VAR', _hypers=None):
        self.metric = _metric(x)
        if _hypers is None:
            self.hypers = Hypers.FlatExp(name)
        else:
            self.hypers = _hypers

    def __call__(self, x1, x2):
        pass

    def cov(self, x1, x2=None):
        pass


class KernelDot(Kernel):
    def __init__(self, x, _metric=ARD_Dot, name='VAR', _hypers=None):
        super().__init__(x, _metric, name, _hypers)

    def __call__(self, x1, x2):
        return self.hypers*self.metric(x1, x2)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.hypers*self.metric.gram(x1, x1)
        else:
            return self.hypers*self.metric.gram(x1, x2)


class LIN(KernelDot):
    def __init__(self, x, _metric=ARD_DotBias, name='VAR', _hypers=None):
        super().__init__(x, _metric, name, _hypers)


class POL(KernelDot):
    def __init__(self, x, p=2, _metric=ARD_DotBias, name='VAR', _hypers=None):
        super().__init__(x, _metric, name, _hypers)
        self.p = p

    def __call__(self, x1, x2):
        return self.hypers*self.metric(x1, x2)**self.p

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.hypers*self.metric.gram(x1, x1)**self.p
        else:
            return self.hypers*self.metric.gram(x1, x2)**self.p


class KernelStationary(Kernel):
    def k(self, d):
        return d

    def __call__(self, x1, x2):
        return self.hypers*self.k(self.metric(x1, x2))

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.hypers*self.k(self.metric.gram(x1, x1))
        else:
            return self.hypers*self.k(self.metric.gram(x1, x2))


class WN(KernelStationary):
    def __init__(self, x=None, _metric=Delta, name='Noise', _hypers=None):
        super().__init__(x, _metric, name, _hypers)

    def __call__(self, x1, x2):
        return self.hypers[0]*self.metric(x1, x2)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.hypers*tt.eye(x1.shape[0])
        else:
            return 0.0


class KernelStationaryExponential(KernelStationary):
    def k(self, d):
        return tt.exp(-d)


class OU(KernelStationaryExponential):
    def __init__(self, x, _metric=ARD_L1, name='VAR_OU', _hypers=None):
        super().__init__(x, _metric, name, _hypers)


class SE(KernelStationaryExponential):
    def __init__(self, x, _metric=ARD_L2, name='VAR_SE', _hypers=None):
        super().__init__(x, _metric, name, _hypers)
