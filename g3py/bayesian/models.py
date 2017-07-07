import os
import threading
import _pickle as pickle
import numpy as np
import theano as th
import theano.tensor as tt
import pymc3 as pm
from ..libs import clone, DictObj
from ..libs.tensors import makefn, tt_to_num
from ..libs.plots import figure, plot, show, plot_text, grid2d, plot_2d
from .. import config
from ipywidgets import interact, interact_manual, FloatSlider
import matplotlib.pyplot as plt

Model = pm.Model


def get_model():
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


def transformed_hypers(hypers):
    trans = DictObj()
    for k, v in hypers.items():
        if type(k) is pm.model.TransformedRV:
            trans[k.transformed] = k.transformed.distribution.transform_used.forward(v).eval()
        else:
            trans[k] = v
    return trans


class GraphicalModel:
    """Abstract class used to define a GraphicalModel.

    Attributes:
        model (pm.Model): Reference to the context pm.Model
    """
    active = None

    def __init__(self, name='GM', description=None, file=None, reset=False):
        if file is not None and not reset:
            try:
                self.reset(file)
                self.activate()
                return
            except:
                print('Not found model in '+str(file))
        # Name, Description, Factor Graph, Space, Hidden
        self.name = name
        if description is None:
            self.description = ''
        else:
            self.description = description
        self.model = get_model()

        self.components = DictObj()
        # Model Average
        self.current_params = None
        self.current_sample = None

        self.fixed_keys = []
        #self.fixed_sample = None
        self.sampling_dims = None
        self.fixed_dims = None
        #self.calc_dimensions() BUG de PYMC3

        if file is not None:
            self.file = file
            try:
                self.save()
            except:
                print('Error in file '+str(file))
        self.activate()

    def activate(self):
        type(self).active = self

    def add_component(self, component):
        self.components[component.name] = component

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

    @property
    def bijection(self):
        return pm.DictToArrayBijection(pm.ArrayOrdering(pm.inputvars(self.model.cont_vars)), self.model.test_point)

    @property
    def ndim(self):
        return self.model.bijection.ordering.dimensions

    def array_to_dict(self, params):
        return self.bijection.rmap(params)

    def dict_to_array(self, params):
        return self.bijection.map(params)

    def set_params(self, params=None):
        self.current_params = params

    def set_sample(self, sample=None):
        self.current_sample = sample

    @property
    def params(self):
        if self.current_params is not None:
            return clone(self.current_params)
        else:
            return self.params_default

    @property
    def params_test(self):
        return clone(self.model.test_point)

    @property
    def params_default(self):
        default = self.params_test
        for name, component in self.components.items():
            for k, v in transformed_hypers(component.default_hypers()).items():
                if k in self.model.vars:
                    default[k.name] = v
        return default

    def params_random(self, mean=None, sigma=0.1, prop=True):
        if mean is None:
            mean = self.params_default
        for k, v in mean.items():
            if prop:
                mean[k] = v * (1 + sigma * np.random.randn(v.size).reshape(v.shape)).astype(th.config.floatX)
            else:
                mean[k] = v + sigma * np.random.randn(v.size).reshape(v.shape).astype(th.config.floatX)
        return mean

    def params_datatrace(self, dt, loc):
        return self.model.bijection.rmap(dt.loc[loc])

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

    def fix_params(self, fixed_params=None):
        if fixed_params is None:
            fixed_params = DictObj()
        self.params_fixed = fixed_params
        self._fixed_keys = list(self.params_fixed.keys())
        self._fixed_array = self.dict_to_array(self.get_params_default())
        self._fixed_chain = None
        self.calc_dimensions()

    @property
    def fixed_vars(self):
        return [t for t in self.model.vars if t.name in self._fixed_keys]

    @property
    def sampling_vars(self):
        return [t for t in self.model.vars if t not in self.fixed_vars]

    def logp_dict(self, params):
        return self.model.logp_array(self.model.dict_to_array(params))

    def logp_array(self, params):
        return self.model.logp_array(params)

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

    def get_params_sampling(self, params=None):
        if params is None:
            params = self.get_params_current()
        return {k: v for k, v in params.items() if k not in self._fixed_keys}

    def get_point(self, point):
        return {v.name: point[v.name] for v in self.model.vars}

    def point_to_in(self, point):
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


class PlotModel:
    def __init__(self, name=None, description = None):
        #print('PlotModel__init__')
        if name is not None:
            self.name = name
        self.is_observed = False
        self.description = description
        if self.description is None:
            self.description = {'title': self.name,
                                'x': 'x',
                                'y': 'y'}
        self._widget_args = None
        self._widget_kwargs = None
        #self._widget_traces = None
        self.widget_params = None
        #self.params_current = None

    @property
    def params_widget(self):
        if self.widget_params is None:
            return self.params
        return clone(self.widget_params)

    def predict(self):
        pass

    def describe(self, title=None, x=None, y=None, text=None):
        if title is not None:
            self.description['title'] = title
        if title is not None:
            self.description['x'] = x
        if title is not None:
            self.description['y'] = y
        if title is not None:
            self.description['text'] = text

    def plot_space(self, independent=False, observed=False):
        if independent:
            for i in range(self.space.shape[1]):
                figure(i)
                plot(self.order, self.space[:, i])
        else:
            plot(self.order, self.space)
        if self.index is not None and observed:
            if independent:
                for i in range(self.space.shape[1]):
                    figure(i)
                    plot(self.index, self.inputs[:, i], '.k')
            else:
                plot(self.index, self.inputs, '.k')

    def plot_hidden(self, big=None):
        if big is None:
            big = config.plot_big
        if big and self.hidden is not None:
            plot(self.order, self.hidden, linewidth=4, label='Hidden Processes')
        elif self.hidden is not None:
            plot(self.order, self.hidden,  label='Hidden Processes')

    def plot_observations(self, big=None):
        if big is None:
            big = config.plot_big
        if big and self.outputs is not None:
            plot(self.index, self.outputs, '.k', ms=20)
            plot(self.index, self.outputs, '.r', ms=15, label='Observations')
        elif self.outputs is not None:
            plot(self.index, self.outputs, '.k', ms=10)
            plot(self.index, self.outputs, '.r', ms=6, label='Observations')

    def plot(self, params=None, space=None, inputs=None, outputs=None, mean=True, var=False, cov=False, median=False, quantiles=True, noise=True, samples=0, prior=False,
             data=True, big=None, plot_space=False, title=None, loc=1):
        values = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, samples=samples, prior=prior)
        if data:
            self.plot_hidden(big)
        if mean:
            plot(self.order, values['mean'], label='Mean')
        if var:
            plot(self.order, values['mean'] + 2.0 * values['std'], '--k', alpha=0.2, label='4.0 std')
            plot(self.order, values['mean'] - 2.0 * values['std'], '--k', alpha=0.2)
        if cov:
            pass
        if median:
            plot(self.order, values['median'], label='Median')
        if quantiles:
            plt.fill_between(self.order, values['quantile_up'], values['quantile_down'], alpha=0.1, label='95%')
        if noise:
            plt.fill_between(self.order, values['noise_up'], values['noise_down'], alpha=0.1, label='noise')
        if samples > 0:
            plot(self.order, values['samples'], alpha=0.4)
        if title is None:
            title = self.description['title']
        if data and self.is_observed:
            self.plot_observations(big)
        if loc is not None:
            plot_text(title, self.description['x'], self.description['y'], loc=loc)
        if plot_space:
            show()
            self.plot_space()
            plot_text('Space X', 'Index', 'Value', legend=False)

    def widget(self, params=None, model=False, auto=False, *args, **kwargs):
        if params is None:
            params = self.params_widget
        intervals = dict()
        for k, v in params.items():
            v = np.squeeze(v)
            if v > 0.1:
                intervals[k] = FloatSlider(min=0.0, max=2*v, value=v, step=1e-2)
            elif v < -0.1:
                intervals[k] = FloatSlider(min=2*v, max=0.0, value=v, step=1e-2)
            else:
                intervals[k] = FloatSlider(min=-5.0, max=5.0, value=v, step=1e-2)
        self._widget_args = args
        self._widget_kwargs = kwargs
        if model:
            widget_plot = self._widget_plot_model
        else:
            widget_plot = self._widget_plot
        if auto:
            interact(widget_plot, **intervals)
        else:
            interact_manual(widget_plot, **intervals)

    def _check_params_dims(self, params):
        r = dict()
        for k, v in params.items():
            try:
                r[k] = np.array(v, dtype=th.config.floatX).reshape(self.model[k].tag.test_value.shape)
            except KeyError:
                pass
        return r

    def _widget_plot(self, **params):
        self.widget_params = self._check_params_dims(params)
        self.plot(params=self.params_widget, *self._widget_args, **self._widget_kwargs)
        show()

    def _widget_plot_model(self, **params):
        self.widget_params = self._check_params_dims(params)
        self.plot_model(params=self.params_widget, indexs=None, kernel=False, mapping=True, marginals=True,
                        bivariate=False)
        show()








    def plot_distribution(self, index=0, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, prior=False, sigma=4, neval=100, title=None, swap=False, label=None):
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, distribution=True, prior=prior)
        domain = np.linspace(pred._mean - sigma * pred.std, pred._mean + sigma * pred.std, neval)
        dist_plot = np.zeros(len(domain))
        for i in range(len(domain)):
            dist_plot[i] = pred.distribution(domain[i:i + 1])
        if label is None:
            if prior:
                label='prior'
            else:
                label='posterior'
        if label is False:
            label = None
        if title is None:
            title = 'Marginal Distribution y_'+str(self.th_order[index])
        if swap:
            plot(dist_plot, domain, label=label)
            plot_text(title, 'Density', 'Domain y')
        else:
            plot(domain, dist_plot,label=label)
            plot_text(title, 'Domain y', 'Density')

    def plot_distribution2D(self, indexs=[0, 1], params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, prior=False, sigma_1=2, sigma_2=2, neval=33, title=None):
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, distribution=True, prior=prior)
        dist1 = np.linspace(pred._mean[0] - sigma_1 * pred.std[0], pred._mean[0] + sigma_1 * pred.std[0], neval)
        dist2 = np.linspace(pred._mean[1] - sigma_2 * pred.std[1], pred._mean[1] + sigma_2 * pred.std[1], neval)
        xy, x2d, y2d = grid2d(dist1, dist2)
        dist_plot = np.zeros(len(xy))
        for i in range(len(xy)):
            dist_plot[i] = pred.distribution(xy[i])
        plot_2d(dist_plot, x2d, y2d)
        if title is None:
            title = 'Distribution2D'
        plot_text(title, 'Domain y_'+str(self.th_order[indexs[0]]), 'Domain y_'+str(self.th_order[indexs[1]]), legend=False)

    def _widget_plot_trace(self, id_trace):
        self._widget_plot(self._check_params_dims(self._widget_traces[id_trace]))

    def widget_traces(self, traces, chain=0):
        self._widget_traces = traces._straces[chain]
        interact(self._widget_plot_trace, __manual=True, id_trace=[0, len(self._widget_traces) - 1])

    def widget_model(self, params=None):
        if params is None:
            params = self.get_params_widget()
        intervals = dict()
        for k, v in params.items():
            v = np.squeeze(v)
            if v > 0.1:
                intervals[k] = [0, 2*v]
            elif v < -0.1:
                intervals[k] = [2*v, 0]
            else:
                intervals[k] = [-5.00, 5.00]
        interact(self._widget_plot_model, __manual=True, **intervals)


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

