import os
# import threading
import _pickle as pickle
import numpy as np
import pymc3 as pm
import theano as th
import theano.tensor as tt
import matplotlib.pyplot as plt
from ..libs import clone, DictObj, save_pkl, load_pkl
from ..libs.tensors import makefn, tt_to_num
from ..libs.plots import figure, plot, show, plot_text
from .. import config
from ipywidgets import interact, interact_manual, FloatSlider
# from numba import jit

# Build the model from pymc3, which it a container for the model.
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

        #self.th_scalar = tt.scalar(self.name + '_scalar_th', dtype=th.config.floatX)
        #self.th_scalar.tag.test_value = np.float32(1)
        self.th_vector = tt.vector(self.name + '_vector_th', dtype=th.config.floatX)
        self.th_vector.tag.test_value = np.array([0.0, 1.0], dtype=th.config.floatX)

        self.components = DictObj()
        self.transformations = DictObj()
        self.potentials = DictObj()
        # Model Average
        self.current_params = None
        self.fixed_datatrace = None
        self.fixed_chain = None
        self.fixed_keys = []
        self.fixed_dims = []

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
        r = load_pkl(path)
        print('Loaded model ' + path)
        r.activate()
        for k,v in r.components.items():
            try:
                v._compile_methods()
            except Exception as m:
                print(m)
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
        #if sample is not None:
        #    self.set_sample(sample)
        try:
            if os.path.isfile(path):
                os.remove(path)
            with self.model:
                save_pkl(self, path)
            print('Saved model '+path)
        except Exception as details:
            print('Error saving model '+path, details)

    @property
    def bijection(self):
        return pm.DictToArrayBijection(pm.ArrayOrdering(pm.inputvars(self.model.cont_vars)), self.model.test_point)

    @property
    def ndim(self):
        return self.model.ndim

    def array_to_dict(self, params):
        return self.bijection.rmap(params)

    def dict_to_array(self, params):
        return self.bijection.map(params)

    def set_params(self, params=None):
        if params is None:
            self.current_params = None
        else:
            self.current_params = DictObj(params)

    @property
    def params(self):
        if self.current_params is not None:
            return clone(self.current_params)
        else:
            return self.params_default

    @property
    def params_test(self):
        return DictObj(self.model.test_point)

    @property
    def params_default(self):
        default = self.params_test
        for name, component in self.components.items():
            d = transformed_hypers(component.default_hypers())
            for k, v in d.items():
                if k in self.model.vars:
                    default[k.name] = v
        return default

    def params_random(self, mean=None, sigma=0.1, prop=True):
        """
        Calculates a set of random parameters for the process around of the mean. In case that the mean it is
        not given, it uses the default.
        Args:
            mean (g3py.libs.DictObj): A dictionary with the parameters of the process.
            sigma (float): the size of the perturbation (the variance)
            prop (bool): whether the perturbation around the mean is aditive (False) or multiplicative (True)

        Returns:
            Returns a random mean obtained around the given mean (if no mean is given, it uses the default)
        """
        if mean is None:
            mean = self.params_default
        for k, v in mean.items():
            if prop:
                mean[k] = v * (1 + sigma * np.random.randn(v.size).reshape(v.shape)).astype(th.config.floatX)
            else:
                mean[k] = v + sigma * np.random.randn(v.size).reshape(v.shape).astype(th.config.floatX)
        return mean

    def params_datatrace(self, dt, loc=None, iloc=None):
        if loc is not None:
            return DictObj(self.model.bijection.rmap(dt.loc[loc]))
        if iloc is not None:
            return DictObj(self.model.bijection.rmap(dt.iloc[iloc]))
        else:
            return DictObj(self.model.bijection.rmap(dt.mean(axis=0)))

    def params_serie(self, serie):
        return DictObj(self.model.bijection.rmap(serie))

    def compile_components(self, precompile=False):
        th_vars = [self.th_vector]
        for v in self.model.deterministics:
            dist = v.transformed.distribution
            # makefn(th_vars, dist.transform_used.backward(self.th_vector), precompile)
            self.transformations[str(v.transformed)] = th.function(th_vars, dist.transform_used.backward(self.th_vector),
                                                                   allow_input_downcast=True, on_unused_input='ignore')
            # makefn(th_vars, dist.transform_used.forward(self.th_vector), precompile)
            self.transformations[str(v
                                     )] = th.function(th_vars, dist.transform_used.forward(self.th_vector),
                                                       allow_input_downcast=True, on_unused_input='ignore')
        th_vars = self.model.vars
        for pot in self.model.potentials:
            self.potentials[str(pot)] = makefn(th_vars, pot, bijection=None)
            self.potentials['array_' + str(pot)] = self.potentials[str(pot)].clone(self.bijection.rmap)

    def transform_params(self, params, to_dict=True, to_transformed=True, complete=False):
        if not isinstance(params, dict):
            params = self.array_to_dict(params)
        if complete or not to_dict:
            r = DictObj(self.params)
        else:
            r = DictObj()
        if to_transformed:
            for k, v in params.items():
                if k in self.original_to_transformed_names:
                    try:
                        r[self.original_to_transformed_names[k]] = self.transformations[k](v)
                    except:
                        r[self.original_to_transformed_names[k]] = self.transformations[k](np.array([v]))
                else:
                    r[k] = v
        else:
            for k, v in params.items():
                if k in self.transformed_to_original_names:
                    try:
                        r[self.transformed_to_original_names[k]] = self.transformations[k](v)
                    except:
                        r[self.transformed_to_original_names[k]] = self.transformations[k](np.array([v]))
                else:
                    r[k] = v

        if not to_dict:
            r = self.dict_to_array(r)
        return r

    @property
    def original_to_transformed_names(self):
        return {k.name: k.transformed.name for k in self.model.deterministics}

    @property
    def transformed_to_original_names(self):
        return {k.transformed.name: k.name for k in self.model.deterministics}

    def fix_vars(self, datatrace=None, keys=None):
        if datatrace is None or keys is None:
            self.fixed_keys = []
            self.fixed_datatrace = None
            self.fixed_chain = None
            self.fixed_dims = []
        else:
            self.fixed_keys = keys
            self.fixed_datatrace = datatrace.copy()
            self.fixed_chain = (self.fixed_datatrace.values[:, :self.ndim]).copy()
            self.fixed_dims = sorted([self.fixed_datatrace.columns.get_loc(k) for k in keys])

    @property
    def sampling_dims(self):
        return sorted(list(set(range(self.ndim)) - set(self.fixed_dims)))

    def sampling_params(self, params):
        if isinstance(params, dict):
            return self.dict_to_array(params)[self.sampling_dims]
        else:
            return params[self.sampling_dims]

    def dict_from_sampling_array(self, params):
        if self.fixed_datatrace is None:
            return self.array_to_dict(params)
        r = self.dict_to_array(self.params)
        r[self.sampling_dims] = params
        return self.array_to_dict(r)

    def eval_th(self, params):
        r = dict()
        for k, v in params.items():
            r[self.model[k]] = v
        return r

# TODO: Check
class OldGraphicalModel:

    def __init__(self):
        self.current_sample = None
        self.sampling_dims = None

        #self.calc_dimensions() BUG de PYMC3

    #TODO: Revisar
    def set_sample(self, sample=None):
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

    def eval_default(self):
        return self.eval_point(self.get_params_default())

    def eval_current(self):
        return self.eval_point(self.get_params_current())


    #TODO: Revisar

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
        return DictObj(self.widget_params)

    def predict(self):
        pass

    def sample(self, params=None, space=None, inputs=None, outputs=None, samples=1, prior=False, noise=False):
        S = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=False, std=False, var=False, cov=False,
                         median=False, quantiles=False, quantiles_noise=False, samples=samples, prior=prior, noise=noise)
        return S['samples']

    def scores(self, params=None, space=None, hidden=None, inputs=None, outputs=None, logp=False, logpred=False, bias=True, variance=False, median=False, *args, **kwargs):
        if hidden is None:
            hidden = self.hidden
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=True, var=variance, median=median, distribution=logpred)
        scores = DictObj()
        if bias:
            scores['_l1'] = np.mean(np.abs(pred.mean - hidden))
            scores['_l2'] = np.mean((pred.mean - hidden)**2)
        if variance:
            scores['_mse'] = np.mean((pred.mean - hidden) ** 2 + pred.variance)
            scores['_rmse'] = np.sqrt(scores['_mse'])
        if median:
            scores['_median_l1'] = np.mean(np.abs(pred.median - hidden))
            scores['_median_l2'] = np.mean((pred.median - hidden)**2)
        if logp:
            scores['_logp'] = self.logp(params)
            scores['_loglike'] = self.loglike(params)
            scores['_logprior'] = self.logp(params, prior=True)
        if logpred:
            scores['_nlpd'] = - pred.logpredictive(hidden) / len(hidden)
        return scores

    def filter_params(self, params):
        #return {k: v for k, v in params.items() if k in self.params}
        return {k: params[k] for k in self.model.test_point}

    def eval_params(self, params=None):
        """
        This function evaluates the params to obtain the l1 and l2 error, and the loglikelihood as well.
        Args:
            params (g3py.libs.DictObj):  The hyperparams came in the dictionary of the params
        Returns:
            It returns the evaluation over the error l1, l2 and the loglikelihood
        """
        r = params.copy()
        r['_ll'] = self.logp(self.filter_params(params))
        r.update(self.scores(params))
        r.update(self.transform_params(r, to_transformed=False))
        return r

    def average(self, datatrace, scores=True, *args, **kwargs):
        """
        For each set of parameters (rows) of the datatrace, the prediction is calculated and then
        calculates the average of the curves.
        Args:
            datatrace (pandas.core.frame.DataFrame): result of MCMC run on DataFrame format. This contains the whole
            information of the evolution of the MCMC.
            scores (bool): If True, error l1 and l2 are calculated and returned.
            *args: List arguments inherited from '.predict' and '.scores' methods.
            **kwargs: Dictionary of arguments inherited from '.predict' and '.scores' methods.

        Returns:
            Returns a dictionary that contains the average prediction, and the scores (if scores True).
            Depending on the *args and **kwargs, the average of the others curves generated from the
            '.predict' method are returned.
        """
        average = None
        for k, v in datatrace.iterrows():
            params = self.active.model.bijection.rmap(v)
            pred = self.predict(params, *args, **kwargs)
            if scores:
                pred.update(self.scores(params, *args, **kwargs))
            if average is None:
                average = pred
            else:
                for k in pred.keys():
                    average[k] += pred[k]
        n = len(datatrace)
        for k in pred.keys():
            average[k] /= n
        return average

    def particles(self, datatrace, nsamples = None, *args, **kwargs):
        """
        Calculates a sample of a gaussian process using the parameters contained in a datatrace.
        Args:
            datatrace (pandas.core.frame.DataFrame): result of MCMC run on DataFrame format. This contains the whole
            information of the evolution of the MCMC.
            nsamples (int): the number of desired samples
            *args: list of arguments that is inherited from the method '.sample'
            **kwargs: dictionary of arguments that is inherithed from the method '.sample'.
        Returns:
            It returns a numpy.ndarray that contains the values of the particles.
        """
        particles = []
        if nsamples is None:
            nsamples = len(datatrace)
        while nsamples > 0:
            for k, v in datatrace.iterrows():
                particles.append(self.sample(self.active.model.bijection.rmap(v), *args, **kwargs))
                nsamples -= 1
                if not nsamples > 0:
                    break
        particles = np.concatenate(particles, axis=1)
        return particles

    def describe(self, title=None, x=None, y=None, text=None):
        """
        Set descriptions to the model
        Args:
            title (str): The title of the process
            x (str):the label for the dependient variable
            y (str):the label for the independient variable
            text (str): aditional text if necessary.
        """

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

    def plot_hidden(self, order=None, hidden=None, big=None):
        if order is None:
            order = self.order
        if hidden is None:
            hidden = self.hidden
        if big is None:
            big = config.plot_big
        if big and hidden is not None:
            plot(order, hidden, 'w', alpha=1.0, lw=4, label='')
            plot(order, hidden, 'k', alpha=0.9, lw=3, label='Hidden Process')
        elif hidden is not None:
            plot(order, hidden, 'w', alpha=0.8, lw=3, label='')
            plot(order, hidden,  'k', alpha=1.0, lw=2, label='Hidden Process')

    def plot_observations(self, index=None, outputs=None, big=None):
        if index is None:
            index = self.index
        if outputs is None:
            outputs = self.outputs
        if big is None:
            big = config.plot_big
        if big and outputs is not None:
            #plot(self.index, self.outputs, 'ow', ms=10, alpha=0.9, label='Observations')
            plot(index, outputs, 'Xw', ms=12)
            plot(index, outputs, 'Xk', ms=10, label='Observations')
        elif outputs is not None:
            #plot(self.index, self.outputs, 'ow', ms=5, alpha=0.8, label='Observations')
            plot(index, outputs, 'Xw', ms=12)
            plot(index, outputs, 'Xk', ms=10, label='Observations')

    def plot(self, params=None, space=None, inputs=None, outputs=None, hidden=True, order=None, mean=True, std=False, cov=False,
             median=False, quantiles=True, quantiles_noise=True, samples=0, palette="Reds", prior=False, noise=False, simulations=100,
             values=None, data=True, logp=True, big=None, plot_space=False, title=None, labels={}, loc='best', ncol=3):
        """
        Creates the prediction of a gp and plot it
        Args:
            params (dict): Contains the hyperparameters of the stochastic process
            space (numpy.ndarray): the domain space of the process
            inputs (numpy.ndarray): the inputs of the process
            outputs (numpy.ndarray): the outputs (observations) of the process
            hidden (bool): Determines whether the hidden process is plotted
            order (numpy.ndarray): For multidimensional process, the order indicates the order in
                which the domain (space) is plotted.
            mean (bool): Determines whether the mean is displayed
            std (bool): Determines whether the standard deviation is displayed
            var (bool): Determines whether the variance is displayed
            cov (bool): Determines whether the covariance is displayed
            median (bool): Determines whether the median is displayed
            quantiles (bool): Determines whether the quantiles (95% of confidence) are displayed
            quantiles_noise (bool): Determines whether the noise is considered for calculating (the
                quantile and it is displayed
            samples (int): the number of samples of the stochastic process that are generated
            palette (str): the name of the color palette
            prior (bool): whether the prediction considers the prior
            noise (bool): wheter the prediction considers noise
            simulations (int): the number of simulation for the aproximation of the value of the
            values (dict): a dictionary that contains the numpy.ndarrays for plotting
            data (bool): If it is True, it will show the values
            logp (bool): If it is True, the tittle shows the value of the log p given the parameters
            big (bool): The default is None. Its value determines the style of the plot. When it is True
                (or none) shows the big style
            plot_space (bool): Whether the space its plotted.
            title (str): The tittle for the figure
            labels (dict): a dictionary that contains the labels of the plotted curves
            loc (str): The location for the legend
            ncol (int): The number of columns of the legend

        Returns:
            It returns a figure that contains the information required in the arguments.
        """
        if values is None:
            values = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, std=std,
                                  cov=cov, median=median, quantiles=quantiles, quantiles_noise=quantiles_noise,
                                  samples=samples, prior=prior, noise=noise, simulations=simulations)

        cmap = plt.get_cmap(palette)
        percs = np.linspace(51, 99, 40)
        colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))

        if order is None:
            order = self.order
        if space is None:
            space = self.space

        if len(order) != len(space):
            if len(space.shape) == 1:
                order = space
            elif space.shape[1] == 1:
                order = space[:, 0]
            else:
                order = np.arange(len(space))
        if samples > 0 and palette is None:
            if 'samples' not in labels:
                labels['samples'] = None
            plot(order, values['samples'][:, 0], alpha=0.4, label=labels['samples'])
            plot(order, values['samples'], alpha=0.4)
        elif samples > 0 and palette is not None:
            if 'samples' not in labels:
                labels['samples'] = None
            plot(order, values['samples'][:, 0], color=cmap(0.9), alpha=0.15, linewidth='1.0', label=labels['samples'])
            plot(order, values['samples'], color=cmap(0.9), alpha=0.15, linewidth='1.0')
        if mean:
            if 'mean' not in labels:
                labels['mean'] = 'Mean'
            plot(order, values['mean'], '-w', alpha=1.0, lw=4)
            #plot(order, values['mean'], '-k',  lw=4)
            plot(order, values['mean'],  '-', color=cmap(1.0), alpha=0.8, lw=3, label=labels['mean'])
        if median:
            if 'median' not in labels:
                labels['median'] = 'Median'
            plot(order, values['median'], '--w', alpha=1.0, lw=4)
            #plot(order, values['median'], '--k', lw=4)
            plot(order, values['median'], '--', color=cmap(1.0), alpha=0.8, lw=3, label=labels['median'])
        if quantiles:
            if 'quantiles' not in labels:
                labels['quantiles'] = '95% CI'
            plot(order, values['quantile_up'], '--', color=cmap(1.0), alpha=0.5, lw=2, label=labels['quantiles'])
            plot(order, values['quantile_down'], '--', color=cmap(1.0), alpha=0.5, lw=2)
            plt.fill_between(order, values['quantile_up'], values['quantile_down'], color=cmap(1.0), alpha=0.1)
        if quantiles_noise:
            if 'quantiles_noise' not in labels:
                labels['quantiles_noise'] = '95% CI + Noise'
            plt.fill_between(order, values['noise_up'], values['noise_down'],  color=cmap(1.0), alpha=0.1, label=labels['quantiles_noise'] )

        if std:
            if 'std' not in labels:
                labels['std'] = '4.0 Std'
            plot(order, values['mean'] + 2.0 * values['std'], '--k', alpha=0.2, label=labels['std'])
            plot(order, values['mean'] - 2.0 * values['std'], '--k', alpha=0.2)
        if cov:
            pass

        if data and hidden is not False:
            self.plot_hidden(big=big)
        if data and self.is_observed:
            self.plot_observations(index=inputs, outputs=outputs, big=big)
        if title is None:
            title = self.description['title']
        if logp:
            if params is None:
                params = self.params
            title += ' (logp: {0:.3f})'.format(np.float32(self.logp(params)))
        if loc is not None:
            plot_text(title, self.description['x'], self.description['y'], loc=loc, ncol=ncol)
        if plot_space:
            show()
            plot(order, space)
            plot_text('Space X', 'Index', 'Value', legend=False)

    def plot_datatrace(self, datatrace, overlap=False, limit=10, scores=True, *args, **kwargs):
        """
        This method takes the hyperparameters of the gp from a datatrace, and plot the prediction
        of each candidate included in the datatrace.
        Args:
            datatrace (pandas.core.frame.DataFrame): result of MCMC run on DataFrame format. This contains the whole
            information of the evolution of the MCMC.
            overlap (bool): If True, the plots are overlaped. If False, are plotted separately.
            limit (int): The maximum number of process plotted.
            scores (bool): If True, the scores l1 and l2 errors are calculated and plotted in the
                title of the figures. If False, the tittle only contains the name of the candidated
                obtained from the datatrace.
            *args: list arguments inherited from the methods '.plot' and '.scores').
            **kwargs: dictionary arguments inherited from the methods '.plot' and '.scores').
        """
        for k, v in datatrace.iterrows():
            self.plot(self.active.model.bijection.rmap(v), *args, **kwargs)
            if not overlap:
                if scores:
                    name = str(k)+' - '+str(self.scores(self.active.model.bijection.rmap(v), *args, **kwargs))
                else:
                    name = str(k)
                plot_text(name, self.description['x'], self.description['y'])
                show()
            if limit is None:
                pass
            elif limit > 1:
                limit -= 1
            else:
                break

    def widget(self, params=None, model=False, auto=False, *args, **kwargs):
        """
        Let set the params in a widget and the posterior visualization
        Args:
            params (dict): The hyperparams of the gp
            model (pymc3 model): The model to display
            auto (bool): If the display its automatic of it is not.
            *args: Variable length argument list, used in the plot function.
            **kwargs: Arbitrary keyword arguments, used in the plot function.
        """
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
