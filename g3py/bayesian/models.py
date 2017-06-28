import os
import threading
import _pickle as pickle
import numpy as np
import theano as th
import theano.tensor as tt
import pymc3 as pm
from ..libs.tensors import makefn, tt_to_num
from ..libs.plots import plot


Model = pm.Model


class GraphicalModel:
    """Abstract class used to define a GraphicalModel.

    Attributes:
        model (pm.Model): Reference to the context pm.Model
    """
    active = None
    def __init__(self, name=None, description=None, file=None, reset=False, precompile=False):
        if file is not None and not reset:
            try:
                self.reset(file)
                self.activate()
                return
            except:
                print('Not found model in '+str(file))
        # Name, Description, Factor Graph, Space, Hidden
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        if description is None:
            self.description = ''
        else:
            self.description = description
        self.model = self.get_model()

        # Model Average
        self.current_sample = None
        self.fixed_keys = []
        self.fixed_sample = None
        self.sampling_dims = None
        self.fixed_dims = None
        self.calc_dimensions()

        if file is not None:
            self.file = file
            try:
                self.save()
            except:
                print('Error in file '+str(file))
        self.activate()

    def activate(self):
        type(self).active = self

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            r = pickle.load(f)
            print('Loaded model ' + path)
        r.activate()
        return r

    def reset(self, path=None):
        if path is None:
            path = self.file
        load = self.load(path)
        self.__dict__.update(load.__dict__)
        self.activate()
        return self

    def save(self, path=None, sample=None):
        if path is None:
            path = self.file
        if sample is not None:
            self.set_sample(sample)
        try:
            if os.path.isfile(path):
                os.remove(path)
            with self.model:
                with open(path, 'wb') as f:
                    pickle.dump(self, f, protocol=-1)
            print('Saved model '+path)
        except Exception as details:
            print('Error saving model '+path, details)

    def get_model(self):
        try:
            model = pm.Model.get_context()
        except:
            model = pm.Model()

        def dlogp(self, vars=None):
            """Nan Robust dlogp"""
            return self.model.fn(tt_to_num(pm.gradient(self.logpt, vars)))

        def fastdlogp(self, vars=None):
            """Nan Robust fastdlogp"""
            return self.model.fastfn(tt_to_num(pm.gradient(self.logpt, vars)))

        def fastd2logp(self, vars=None):
            """Nan Robust fastd2logp"""
            return self.model.fastfn(tt_to_num(-pm.jacobian(tt_to_num(pm.gradient(self.logpt, vars), vars))))

        import types
        model.dlogp = types.MethodType(dlogp, model)
        model.fastdlogp = types.MethodType(fastdlogp, model)
        model.fastd2logp = types.MethodType(fastd2logp, model)

        return model

    def set_sample(self, sample):
        self.current_sample = sample

    def fix_vars(self, keys=[], sample=None):
        self.fixed_keys = keys
        self.fixed_sample = sample
        self.calc_dimensions()

    def calc_dimensions(self):
        dimensions = list(range(self.ndim))
        dims = list()
        for k in self.model.bijection.ordering.vmap:
            if k.var not in self.fixed_keys:
                dims += dimensions[k.slc]
        self.sampling_dims = dims
        dims = list()
        for k in self.model.bijection.ordering.vmap:
            if k.var in self.fixed_keys:
                dims += dimensions[k.slc]
        self.fixed_dims = dims

    @property
    def fixed_vars(self):
        return [t for t in self.model.vars if t.name in self._fixed_keys]

    @property
    def sampling_vars(self):
        return [t for t in self.model.vars if t not in self.fixed_vars]

    @property
    def ndim(self):
        return self.model.bijection.ordering.dimensions

    def get_params_model(self, process=None, params=None, current=None, fixed=False):
        if process is None:
            process = self
        if params is None:
            params = process.get_params_current()
        if current is None:
            current = self.get_params_current()
        params_transform = {k.replace(process.name, self.name, 1): v for k, v in params.items()}
        params_return = DictObj({k: v for k, v in params_transform.items() if k in current.keys()})
        params_return.update({k: v for k, v in current.items() if k not in params_transform.keys()})
        if fixed:
            params_return.update(self.params_fixed)
        return params_return

    def get_params_random(self, mean=None, sigma=0.1, prop=True, fixed=True):
        if mean is None:
            mean = self.get_params_default()
        for k, v in mean.items():
            if prop:
                mean[k] = v * (1 + sigma * np.random.randn(v.size).reshape(v.shape)).astype(th.config.floatX)
            else:
                mean[k] = v + sigma * np.random.randn(v.size).reshape(v.shape).astype(th.config.floatX)
        if fixed:
            mean.update(self.params_fixed)
        return mean

    def get_params_test(self, fixed=False):
        test = clone(self.model.test_point)
        if fixed:
            test.update(self.params_fixed)
        return test

    def get_params_default(self, fixed=True):
        if self.observed_index is None:
            return self.get_params_test(fixed)
        default = self.get_params_test(fixed)
        for k, v in trans_hypers(self.default_hypers()).items():
            if k in self.model.vars:
                default[k.name] = v
        if fixed:
            default.update(self.params_fixed)
        return default

    def get_params_current(self, fixed=True):
        if self.params_current is None:
            return self.get_params_default(fixed)
        if fixed:
            self.params_current.update(self.params_fixed)
        return clone(self.params_current)

    def get_params_widget(self, fixed=False):
        if self.params_widget is None:
            return self.get_params_default(fixed)
        if fixed:
            self.params_widget.update(self.params_fixed)
        return clone(self.params_widget)

    def get_params_sampling(self, params=None):
        if params is None:
            params = self.get_params_current()
        return {k: v for k, v in params.items() if k not in self._fixed_keys}

    def get_params_datatrace(self, dt, loc):
        return self.model.bijection.rmap(dt.loc[loc])

    def get_point(self, point):
        return {v.name: point[v.name] for v in self.model.vars}

    def point_to_In(self, point):
        r = list()
        for k, v in point.items():
            r.append(th.In(self.model[k], value=v))
        return r

    def eval_point(self, point):
        r = dict()
        for k, v in point.items():
            r[self.model[k]] = v
        return r

    def eval_default(self):
        return self.eval_point(self.get_params_default())

    def eval_current(self):
        return self.eval_point(self.get_params_current())

    def eval_widget(self):
        return self.eval_point(self.get_params_widget())



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

