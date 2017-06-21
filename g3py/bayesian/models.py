import _pickle as pickle
import numpy as np
import theano as th
import theano.tensor as tt
import pymc3 as pm
from ..libs.tensors import makefn
from ..libs.plots import plot


Model = pm.Model


def load_model(path):
    # with pm.Model():
    with open(path, 'rb') as f:
        r = pickle.load(f)
        print('Loaded model ' + path)
        return r


class GraphicalModel(pm.Model):
    pass


class BlackBox(object):
    def __new__(cls, *args, **kwargs):
        instance = super(BlackBox, cls).__new__(cls)
        instance.__init__(*args, **kwargs)
        try:
            instance._predict = instance.compile()
            print(cls.__name__, ':', instance.hypers_ancestors(), '-->', instance.symbolic(), '\n')
        except Exception as e:
            print(cls.__name__, ':', instance.hypers_ancestors(), '-->', instance.symbolic(),
                  '\nCompilation Error!', e, '\n')
        return instance

    def compile(self):
        return makefn(self.hypers_ancestors(), self.symbolic(), precompile=True)

    def inputs(self):
        return []

    def hypers(self):
        return []

    def _hypers_ancestors(self):
        hypers_return = []
        for p in self.inputs():
            if hasattr(p, 'hypers'):
                hypers_return += p.hypers_ancestors()
        return hypers_return

    def hypers_ancestors(self):
        return list(set(self.hypers() + self._hypers_ancestors()))

    def symbolic(self):
        pass

    def logp(self, params=None):
        return np.float32(0.0)

    def predict(self, datatrace=None, safe=True):
        if datatrace is None:
            return self._predict()
        else:
            if safe:
                datatrace = {k: v for k, v in datatrace.items() if k in [v.name for v in self.hypers_ancestors()]}
            return self._predict(**(datatrace))

    def plot(self, datatrace=None, prediction=None, fig=1, subplot=111):
        if prediction is None:
            prediction = self.predict(datatrace)
        plot(prediction.T)

    def eval(self, symb=None):
        if symb is None:
            symb = self.random()
        return self.symbolic().eval(symb)

    def random(self, size=None, index_tensor=False):
        random = {}
        for p in self.hypers_ancestors():
            if index_tensor:
                index = p
            else:
                index = p.name
            if hasattr(p.pymc3, 'distribution'):
                random[index] = p.pymc3.distribution.dist.random(size=size).astype(th.config.floatX)[:, None]
            elif hasattr(p.pymc3, 'random'):
                random[index] = p.pymc3.random(size=size).astype(th.config.floatX)[:, None]
        # for p in self._hypers_ancestors():
        #    random[p]=p.random()
        return random

    @property
    def type(self):
        return self.symbolic().type

    @property
    def shape(self):
        return self.eval().shape


class TheanoBlackBox(BlackBox):
    pass


class NumpyBlackBox(BlackBox):
    pass


class ScalingQuantiles(TheanoBlackBox):
    def __init__(self, name, serie, quantile):
        self.serie = th.shared(np.float32(serie), name=name, borrow=True, allow_downcast=True)
        self.quantile = np.float32(np.percentile(serie, quantile))

    def symbolic(self):
        return self.serie*self.quantile

    def random(self):
        return {self.serie.name:self.serie.get_value()}


class BlackBoxSums(TheanoBlackBox):
    def __init__(self, *sums):
        self.sums = list(sums)

    def inputs(self):
        return self.sums

    def hypers(self):
        return []

    def symbolic(self):
        r = np.float32(0.0)
        for s in self.sums:
            r += s.symbolic()
        return r


class BlackBoxAR1(TheanoBlackBox):
    def __init__(self, parent, decay, init):
        self.parent = parent
        self.decay = decay
        self.init = init

    def inputs(self):
        return [self.parent]

    def hypers(self):
        return [self.decay, self.init]

    def symbolic(self):
        return ar1_batch(self.parent.symbolic(), tt.exp(-self.decay), self.init)


def uniform(name, lower=0.0, upper=1.0, testval=None):
    if testval is None:
        testval = (lower + upper) / 2
    dist = pm.Uniform(name, testval=np.float32(testval), lower=np.float32(lower),
                      upper=np.float32(upper), shape=(1), dtype=th.config.floatX)
    distn = dist.dimshuffle([0, 'x'])
    distn = tt.unbroadcast(distn, 0)
    distn.name = name
    distn.pymc3 = dist
    return distn


def ar1(s, d, i=0):
    s = tt.inc_subtensor(s[:, :1], d * i / (1 - d))
    time = tt.arange(0, s.shape[1], dtype=th.config.floatX)
    time_2 = (s.shape[1] // 2).astype(dtype=th.config.floatX)
    d_time = d ** (time_2 - time)
    d_ads = (1 - d) * tt.cumsum(s * d_time, axis=1) / d_time
    return d_ads


def ar1_batch(s, d, i=0, batch=100, steps=5):
    r0 = ar1(s[:, :batch], d, i)
    r = [r0]
    for k in range(1, steps):
        r0 = ar1(s[:, k * batch:(k + 1) * batch], d, r0[:, -1:])
        r += [r0]
    return tt.concatenate(r, axis=1)

