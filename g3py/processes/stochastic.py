import _pickle as pickle
import os
import time
import datetime
import numpy as np
import pymc3 as pm
import scipy as sp
import theano as th
from ..functions import Mean, Kernel, Mapping, KernelSum, WN, tt_to_num, def_space, trans_hypers
from ..libs import tt_to_cov, makefn, plot_text, clone, DictObj, plot_2d, grid2d, show, nan_to_high, MaxTime
from ..models import ConstantStep, RobustSlice
from .. import config
from ipywidgets import interact
from matplotlib import pyplot as plt
from theano import tensor as tt
from scipy import optimize
from inspect import signature
from tqdm import tqdm
import theano.tensor.nlinalg as nL
import theano.tensor.slinalg as sL
import theano.tensor.slinalg as tsl
import theano.tensor.nlinalg as tnl
import emcee

Model = pm.Model

def load_model(path):
    # with pm.Model():
    with open(path, 'rb') as f:
        #try:
        r = pickle.load(f)
        print('Loaded model ' + path)
        return r
        #except:
        #    print('Error loading model '+path)


class StochasticProcess:
    """Abstract class used to define a StochasticProcess.

    Attributes:
        model (pm.Model): Reference to the context pm.Model
    """
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, mapping: Mapping=None, noise=True, freedom=None,
                 name=None, inputs=None, outputs=None, hidden=None, description=None, file=None, recompile=False, precompile=False):
        if file is not None and not recompile:
            try:
                load = load_model(file)
                self.__dict__.update(load.__dict__)
                return
            except:
                print('Not found model in '+str(file))
        # Name, Description, Factor Graph, Space, Hidden
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        if description is None:
            self.description = {'title': 'title',
                                'x': 'x',
                                'y': 'y',
                                'text': 'text'}
        else:
            self.description = description
        self.model = self.get_model()

        # Space, Hidden, Observed
        if type(space) is int:
            space = np.random.rand(2, space).astype(dtype=th.config.floatX)
        space_raw = space
        space = space_raw[:2]
        __, self.space_values, self.space_index = def_space(space)
        self.inputs, self.inputs_values, self.observed_index = def_space(space, self.name + '_inputs')
        self.outputs, self.outputs_values, __ = def_space(np.zeros(len(space)), self.name + '_outputs', squeeze=True)

        self.space_th = tt.matrix(self.name + '_space_th', dtype=th.config.floatX)
        self.inputs_th = tt.matrix(self.name + '_inputs_th', dtype=th.config.floatX)
        self.outputs_th = tt.vector(self.name + '_outputs_th', dtype=th.config.floatX)
        self.random_th = tt.vector(self.name + '_random_th', dtype=th.config.floatX)
        self.random_scalar = tt.scalar(self.name + '_random_scalar', dtype=th.config.floatX)

        self.space_th.tag.test_value = self.space_values
        self.inputs_th.tag.test_value = self.inputs_values
        self.outputs_th.tag.test_value = self.outputs_values
        self.random_th.tag.test_value = np.random.randn(len(self.space_values)).astype(dtype=th.config.floatX)
        self.random_scalar.tag.test_value = np.float32(10)

        if hidden is None:
            self.hidden = None
        else:
            self.hidden = np.squeeze(hidden)

        # Parametrization
        self.location = location
        if noise:
            self.kernel_f = kernel
            self.kernel = KernelSum(self.kernel_f, WN(self.space_values, 'Noise'))
        else:
            self.kernel_f = kernel
            self.kernel = self.kernel_f
        self.mapping = mapping
        self.freedom = freedom

        with self.model:
            self.location.check_dims(self.inputs_values)
            self.kernel.check_dims(self.inputs_values)
            self.mapping.check_dims(self.outputs_values)

            self.location.check_hypers(self.name + '_')
            self.kernel.check_hypers(self.name + '_')
            self.mapping.check_hypers(self.name + '_')

            if self.freedom is not None:
                self.freedom.check_dims(None)
                self.freedom.check_hypers(self.name + '_')

        print('Space Dimension: ', self.space_values.shape[1])

        # Hyper-parameters values
        self._widget_traces = None
        self.params_current = None
        self.params_widget = None
        self.params_fixed = DictObj()

        # Basic Tensors
        self.location_space = self.location(self.space_th)
        self.location_inputs = self.location(self.inputs)

        self.kernel_space = tt_to_cov(self.kernel.cov(self.space_th))
        self.kernel_inputs = tt_to_cov(self.kernel.cov(self.inputs))
        self.kernel_space_inputs = tt_to_num(self.kernel.cov(self.space_th, self.inputs))

        self.kernel_f_space = tt_to_cov(self.kernel_f.cov(self.space_th))
        self.kernel_f_inputs = tt_to_cov(self.kernel_f.cov(self.inputs))
        self.kernel_f_space_inputs = tt_to_num(self.kernel_f.cov(self.space_th, self.inputs))

        self.mapping_outputs = tt_to_num(self.mapping.inv(self.outputs))
        self.mapping_th = tt_to_num(self.mapping(self.random_th))
        self.mapping_inv_th = tt_to_num(self.mapping.inv(self.random_th))

        # Prior
        self.prior_mean = None
        self.prior_covariance = None
        self.prior_cholesky = None
        self.prior_logp = None
        self.prior_logpred = None
        self.prior_distribution = None
        self.prior_variance = None
        self.prior_std = None
        self.prior_noise = None
        self.prior_median = None
        self.prior_quantile_up = None
        self.prior_quantile_down = None
        self.prior_noise_up = None
        self.prior_noise_down = None
        self.prior_sampler = None

        # Posterior
        self.posterior_mean = None
        self.posterior_covariance = None
        self.posterior_cholesky = None
        self.posterior_logp = None
        self.posterior_logpred = None
        self.posterior_distribution = None
        self.posterior_variance = None
        self.posterior_std = None
        self.posterior_noise = None
        self.posterior_median = None
        self.posterior_quantile_up = None
        self.posterior_quantile_down = None
        self.posterior_noise_up = None
        self.posterior_noise_down = None
        self.posterior_sampler = None

        self.distribution = None

        self.compiles = DictObj()
        print('Init Definition')
        self.define_process()
        print('Definition OK')

        self.compile(precompile)
        print('Compilation OK')

        self.observed(inputs, outputs)

        self.logp_prior = None
        self.compile_logprior()

        #__, self.space_values, self.space_index = def_space(space_raw)
        self.set_space(space_raw, self.hidden)
        if file is not None:
            self.file = file
            try:
                self.save_model()
            except:
                print('Error in file '+str(file))

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

    def compile_logprior(self):
        self.logp_prior = self.model.bijection.mapf(self.model.fn(tt.add(*map(tt.sum, [var.logpt for var in self.model.free_RVs] + self.model.potentials))))

    def logp_like(self, *args, **kwargs):
        return self.model.logp_array(*args, **kwargs) - self.logp_prior(*args, **kwargs)

    def logp_array(self, params):
        return self.model.logp_array(params)

    def dlogp_array(self, params):
        return self.model.dlogp_array(params)

    def dict_to_array(self, params):
        return self.model.dict_to_array(params)

    def logp_dict(self, params):
        return self.model.logp_array(self.model.dict_to_array(params))

    def define_distribution(self):
        pass

    def define_process(self):
        pass

    def compile(self, precompile=False):
        params = [self.space_th] + self.model.vars
        self.compiles['location_space'] = makefn(params, self.location_space, precompile)
        self.compiles['kernel_space'] = makefn(params, self.kernel_space, precompile)
        self.compiles['kernel_f_space'] = makefn(params, self.kernel_f_space, precompile)

        params = [self.inputs_th] + self.model.vars
        self.compiles['location_inputs'] = makefn(params, self.location_inputs, precompile)
        self.compiles['kernel_inputs'] = makefn(params, self.kernel_inputs, precompile)
        self.compiles['kernel_f_inputs'] = makefn(params, self.kernel_f_inputs, precompile)

        params = [self.space_th, self.inputs_th] + self.model.vars
        self.compiles['kernel_space_inputs'] = makefn(params, self.kernel_space_inputs, precompile)
        self.compiles['kernel_f_space_inputs'] = makefn(params, self.kernel_f_space_inputs, precompile)

        params = self.model.vars
        self.compiles['mapping_outputs'] = makefn(params, self.mapping_outputs, precompile)
        self.compiles['mapping_th'] = makefn([self.random_th] + params, self.mapping_th, precompile)
        self.compiles['mapping_inv_th'] = makefn([self.random_th] + params, self.mapping_inv_th, precompile)

        params = [self.space_th] + self.model.vars
        self.compiles['prior_mean'] = makefn(params, self.prior_mean, precompile)
        self.compiles['prior_covariance'] = makefn(params, self.prior_covariance, precompile)
        self.compiles['prior_cholesky'] = makefn(params, self.prior_cholesky, precompile)
        self.compiles['prior_variance'] = makefn(params, self.prior_variance, precompile)
        self.compiles['prior_std'] = makefn(params, self.prior_std, precompile)
        self.compiles['prior_noise'] = makefn(params, self.prior_noise, precompile)
        self.compiles['prior_median'] = makefn(params, self.prior_median, precompile)
        self.compiles['prior_quantile_up'] = makefn(params, self.prior_quantile_up, precompile)
        self.compiles['prior_quantile_down'] = makefn(params, self.prior_quantile_down, precompile)
        self.compiles['prior_noise_up'] = makefn(params, self.prior_noise_up, precompile)
        self.compiles['prior_noise_down'] = makefn(params, self.prior_noise_down, precompile)
        self.compiles['prior_logp'] = makefn([self.random_th] + params, self.prior_logp, precompile)
        self.compiles['prior_logpred'] = makefn([self.random_th] + params, self.prior_logpred, precompile)
        self.compiles['prior_distribution'] = makefn([self.random_th] + params, self.prior_distribution, precompile)
        try:
            self.compiles['prior_sampler'] = makefn([self.random_th] + params, self.prior_sampler, precompile)
        except:
            self.compiles['prior_sampler'] = makefn([self.random_scalar, self.random_th] + params, self.prior_sampler, precompile)

        params = [self.space_th, self.inputs_th, self.outputs_th] + self.model.vars
        self.compiles['posterior_mean'] = makefn(params, self.posterior_mean, precompile)
        self.compiles['posterior_covariance'] = makefn(params, self.posterior_covariance, precompile)
        self.compiles['posterior_cholesky'] = makefn(params, self.posterior_cholesky, precompile)
        self.compiles['posterior_variance'] = makefn(params, self.posterior_variance, precompile)
        self.compiles['posterior_std'] = makefn(params, self.posterior_std, precompile)
        self.compiles['posterior_noise'] = makefn(params, self.posterior_noise, precompile)
        self.compiles['posterior_median'] = makefn(params, self.posterior_median, precompile)
        self.compiles['posterior_quantile_up'] = makefn(params, self.posterior_quantile_up, precompile)
        self.compiles['posterior_quantile_down'] = makefn(params, self.posterior_quantile_down, precompile)
        self.compiles['posterior_noise_up'] = makefn(params, self.posterior_noise_up, precompile)
        self.compiles['posterior_noise_down'] = makefn(params, self.posterior_noise_down, precompile)
        self.compiles['posterior_logp'] = makefn([self.random_th] + params, self.posterior_logp, precompile)
        self.compiles['posterior_logpred'] = makefn([self.random_th] + params, self.posterior_logpred, precompile)
        self.compiles['posterior_distribution'] = makefn([self.random_th] + params, self.posterior_distribution, precompile)
        try:
            self.compiles['posterior_sampler'] = makefn([self.random_th] + params, self.posterior_sampler, precompile)
        except:
            self.compiles['posterior_sampler'] = makefn([self.random_scalar, self.random_th] + params, self.posterior_sampler, precompile)

    def describe(self, title=None, x=None, y=None, text=None):
        if title is not None:
            self.description['title'] = title
        if title is not None:
            self.description['x'] = x
        if title is not None:
            self.description['y'] = y
        if title is not None:
            self.description['text'] = text

    def set_space(self, space, hidden=None, index=None):
        __, self.space_values, self.space_index = def_space(space)
        self.hidden = hidden
        if index is not None:
            self.space_index = index

    def observed(self, inputs=None, outputs=None, index=None):
        if inputs is None or outputs is None or len(inputs) == 0 or len(inputs) == 0:
            self.inputs_values, self.outputs_values, self.observed_index = None, None, index
            self.inputs.set_value(self.inputs_values, borrow=True)
            self.outputs.set_value(self.outputs_values, borrow=True)
            return

        __, self.inputs_values, self.observed_index = def_space(inputs)
        if index is not None:
            self.observed_index = index
        __, self.outputs_values, __ = def_space(outputs, squeeze=True)
        self.inputs.set_value(self.inputs_values, borrow=True)
        self.outputs.set_value(self.outputs_values, borrow=True)
        if self.distribution is None:
            with self.model:
                self.define_distribution()

    def prior(self, params=None, space=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, samples=0, distribution=False):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        values = DictObj()
        if mean:
            values['mean'] = self.compiles['prior_mean'](space, **params)
        if var:
            values['variance'] = self.compiles['prior_variance'](space, **params)
            values['std'] = self.compiles['prior_std'](space, **params)
        if cov:
            values['covariance'] = self.compiles['prior_covariance'](space, **params)
        if median:
            values['median'] = self.compiles['prior_median'](space, **params)
        if quantiles:
            values['quantile_up'] = self.compiles['prior_quantile_up'](space, **params)
            values['quantile_down'] = self.compiles['prior_quantile_down'](space, **params)
        if noise:
            values['noise'] = self.compiles['prior_noise'](space, **params)
            values['noise_up'] = self.compiles['prior_noise_up'](space, **params)
            values['noise_down'] = self.compiles['prior_noise_down'](space, **params)
        # TODO: if samples is with another distribution
        if samples > 0:
            S = np.empty((len(space), samples))
            rand = np.random.randn(len(space), samples)
            for i in range(samples):
                S[:, i] = self.compiles['prior_sampler'](rand[:, i], space, **params)
                values['samples'] = S
        if distribution:
            values['logp'] = lambda x: self.compiles['prior_logp'](x, space, **params)
            values['logpred'] = lambda x: self.compiles['prior_logpred'](x, space, **params)
            values['distribution'] = lambda x: self.compiles['prior_distribution'](x, space, **params)
        return values

    def posterior(self, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, samples=0, distribution=False):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        if inputs is None:
            inputs = self.inputs_values
        if outputs is None:
            outputs = self.outputs_values
        values = DictObj()
        if mean:
            values['mean'] = self.compiles['posterior_mean'](space, inputs, outputs, **params)
        if var:
            values['variance'] = self.compiles['posterior_variance'](space, inputs, outputs, **params)
            values['std'] = self.compiles['posterior_std'](space, inputs, outputs, **params)
        if cov:
            values['covariance'] = self.compiles['posterior_covariance'](space, inputs, outputs, **params)
        if median:
            values['median'] = self.compiles['posterior_median'](space, inputs, outputs, **params)
        if quantiles:
            values['quantile_up'] = self.compiles['posterior_quantile_up'](space, inputs, outputs, **params)
            values['quantile_down'] = self.compiles['posterior_quantile_down'](space, inputs, outputs, **params)
        if noise:
            values['noise'] = self.compiles['posterior_noise'](space, inputs, outputs, **params)
            values['noise_up'] = self.compiles['posterior_noise_up'](space, inputs, outputs, **params)
            values['noise_down'] = self.compiles['posterior_noise_down'](space, inputs, outputs, **params)
        # TODO: if samples is with another distribution
        if samples > 0:
            S = np.empty((len(space), samples))
            rand = np.random.randn(len(space), samples)
            for i in range(samples):
                S[:, i] = self.compiles['posterior_sampler'](rand[:, i], space, inputs, outputs, **params)
            values['samples'] = S
        if distribution:
            values['logp'] = lambda x: self.compiles['posterior_logp'](x, space, inputs, outputs, **params)
            values['logpred'] = lambda x: self.compiles['posterior_logpred'](x, space, inputs, outputs, **params)
            values['distribution'] = lambda x: self.compiles['posterior_distribution'](x, space, inputs, outputs, **params)
        return values

    def sample(self, params=None, space=None, inputs=None, outputs=None, samples=1, prior=False):
        S = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=False, var=False, cov=False, median=False, quantiles=False, noise=False, samples=samples, prior=False)
        return S['samples']

    def predict(self, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, samples=0, distribution=False, prior=False):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        if inputs is None:
            inputs = self.inputs_values
        if outputs is None:
            outputs = self.outputs_values
        if prior or self.observed_index is None:
            return self.prior(params=params, space=space, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, samples=samples, distribution=distribution)
        else:
            return self.posterior(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, samples=samples, distribution=distribution)

    def scores(self, params=None, space=None, hidden=None, inputs=None, outputs=None, logp=True, bias=True, variance=False, median=False):
        if hidden is None:
            hidden = self.hidden
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, var=variance, median=median, distribution=logp)
        scores = DictObj()
        if bias:
            scores['_BiasL1'] = np.mean(np.abs(pred.mean - hidden))
            scores['_BiasL2'] = np.mean((pred.mean - hidden)**2)
        if variance:
            scores['_MSE'] = np.mean((pred.mean - hidden) ** 2 + pred.variance)
            scores['_RMSE'] = np.sqrt(scores['_MSE'])
        if median:
            scores['_MedianL1'] = np.mean(np.abs(pred.median - hidden))
            scores['_MedianL2'] = np.mean((pred.median - hidden)**2)
        if logp:
            #scores['_NLL'] = - pred.logp(hidden) / len(hidden)
            scores['_NLPD'] = - pred.logpred(hidden) / len(hidden)
        return scores

    def plot(self, params=None, space=None, inputs=None, outputs=None, mean=True, var=False, cov=False, median=False, quantiles=True, noise=True, samples=0, prior=False,
             data=True, big=None, plot_space=False, title=None, loc=1):
        values = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, samples=samples, prior=prior)
        if space is not None:
            self.set_space(space)
        if data:
            self.plot_hidden(big)
        if mean:
            plt.plot(self.space_index, values['mean'], label='Mean')
        if var:
            plt.plot(self.space_index, values['mean'] + 2.0 * values['std'], '--k', alpha=0.2, label='4.0 std')
            plt.plot(self.space_index, values['mean'] - 2.0 * values['std'], '--k', alpha=0.2)
        if cov:
            pass
        if median:
            plt.plot(self.space_index, values['median'], label='Median')
        if quantiles:
            plt.fill_between(self.space_index, values['quantile_up'], values['quantile_down'], alpha=0.1, label='95%')
        if noise:
            plt.fill_between(self.space_index, values['noise_up'], values['noise_down'], alpha=0.1, label='noise')
        if samples > 0:
            plt.plot(self.space_index, values['samples'], alpha=0.4)
        if title is None:
            title = self.description['title']
        if data:
            self.plot_observations(big)
        plot_text(title, self.description['x'], self.description['y'], loc=loc)
        show()
        if plot_space:
            self.plot_space()
            plot_text('Space X', 'Index', 'Value', legend=False)

    def plot_distribution(self, index=0, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, prior=False, sigma=4, neval=100, title=None):
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, distribution=True, prior=prior)
        domain = np.linspace(pred.mean - sigma * pred.std, pred.mean + sigma * pred.std, neval)
        dist_plot = np.zeros(len(domain))
        for i in range(len(domain)):
            dist_plot[i] = pred.distribution(domain[i:i + 1])
        if prior:
            plt.plot(domain, dist_plot, label='prior')
        else:
            plt.plot(domain, dist_plot, label='posterior')
        if title is None:
            title = 'Marginal Distribution'
        plot_text(title+' y_'+str(self.space_index[index]), 'Domain y', '')

    def plot_mapping(self, params=None, space=None, inputs=None, outputs=None, neval=100,title=None):
        if params is None:
            params = self.get_params_current()
        if outputs is None:
            outputs = self.outputs_values
        domain = np.linspace(outputs.min() , outputs.max() , neval)
        transform = self.compiles['mapping_inv_th'](domain, **params)
        plt.plot(domain, transform, label='mapping')

        #inv_domain = np.linspace(transform.min() - transform.std(), transform.max() + transform.std(), neval)
        #inv_transform = self.compiles['mapping_th'](inv_domain, **params)
        #plt.plot(inv_transform, inv_domain, label='mapping_th')
        if title is None:
            title = 'Mapping'
        plot_text(title, 'Domain y', 'Domain T(y)')

    def plot_kernel(self, params=None, space=None, inputs=None, centers=[1/10, 1/2, 9/10]):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        if inputs is None:
            inputs = self.inputs_values
        ksi = self.compiles['kernel_space_inputs'](space, inputs, **params).T
        for ind in centers:
            plt.plot(self.space_index, ksi[int(len(ksi)*ind), :], label='k(x_'+str(int(len(ksi)*ind))+')')
        plot_text('Kernel', 'Space x', 'Kernel value k(x,v)')

    def plot_concentration(self, params=None, space=None):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        plt.matshow(self.compiles['kernel_space'](space, **params))
        plot_text('Concentration', 'Space x', 'Space x')

    def plot_location(self, params=None, space=None):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.space_values
        plt.plot(self.space_index, self.compiles['location_space'](space, **params), label='location')
        plot_text('Location', 'Space x', 'Location value m(x)')

    def plot_distribution2D(self, indexs=[0,1], params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, prior=False, sigma_1=2, sigma_2=2, neval=33, title=None):
        pred = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=mean, var=var, cov=cov, median=median, quantiles=quantiles, noise=noise, distribution=True, prior=prior)
        dist1 = np.linspace(pred.mean[0] - sigma_1 * pred.std[0], pred.mean[0] + sigma_1 * pred.std[0], neval)
        dist2 = np.linspace(pred.mean[1] - sigma_2 * pred.std[1], pred.mean[1] + sigma_2 * pred.std[1], neval)
        xy, x2d, y2d = grid2d(dist1, dist2)
        dist_plot = np.zeros(len(xy))
        for i in range(len(xy)):
            dist_plot[i] = pred.distribution(xy[i])
        plot_2d(dist_plot, x2d, y2d)
        if title is None:
            title = 'Distribution2D'
        plot_text(title, 'Domain y_'+str(self.space_index[indexs[0]]), 'Domain y_'+str(self.space_index[indexs[1]]), legend=False)

    def plot_model(self, params=None, indexs=None, kernel=True, mapping=True, marginals=True, bivariate=True):
        if indexs is None:
            indexs = [self.observed_index[len(self.observed_index)//2], self.observed_index[len(self.observed_index)//2]+1]

        if kernel:
            plt.subplot(121)
            self.plot_kernel(params=params)
        if mapping:
            plt.subplot(122)
            self.plot_mapping(params=params)
        show()

        if marginals:
            plt.subplot(121)
            self.plot_distribution(index=indexs[0], params=params, space=self.space_values[indexs[0]:indexs[0]+1, :], prior=True)
            self.plot_distribution(index=indexs[0], params=params, space=self.space_values[indexs[0]:indexs[0]+1, :])
            plt.subplot(122)
            self.plot_distribution(index=indexs[1], params=params, space=self.space_values[indexs[1]:indexs[1]+1, :], prior=True)
            self.plot_distribution(index=indexs[1], params=params, space=self.space_values[indexs[1]:indexs[1]+1, :])
            show()
        if bivariate:
            self.plot_distribution2D(indexs=indexs, params=params, space=self.space_values[indexs, :])
            show()

    def plot_kernel2D(self):
        pass

    def set_params(self, params):
        self.params_current = params

    def fix_params(self, fixed_params=None):
        if fixed_params is None:
            fixed_params = DictObj()
        self.params_fixed = fixed_params

    @property
    def fixed_vars(self):
        return [t for t in self.model.vars if t.name in self.params_fixed.keys()]

    @property
    def sampling_vars(self):
        return [t for t in self.model.vars if t not in self.fixed_vars]

    def check_params_dims(self, params):
        r = dict()
        for k, v in params.items():
            try:
                r[k] = np.array(v, dtype=th.config.floatX).reshape(self.model[k].tag.test_value.shape)
            except KeyError:
                pass
        return r

    def default_hypers(self):
        x = np.array(self.inputs_values)
        y = np.squeeze(np.array(self.outputs_values))
        return {**self.location.default_hypers_dims(x, y), **self.kernel.default_hypers_dims(x, y),
                **self.mapping.default_hypers_dims(x, y)}

    def widget_plot(self, params):
        self.params_widget = params
        self.plot(params = self.params_widget)

    def widget_traces(self, traces, chain=0):
        self._widget_traces = traces._straces[chain]
        interact(self.widget_plot_trace, __manual=True, id_trace=[0, len(self._widget_traces) - 1])

    def widget_plot_trace(self, id_trace):
        self.widget_plot(self.check_params_dims(self._widget_traces[id_trace]))

    def widget_plot_params(self, **params):
        self.widget_plot(self.check_params_dims(params))

    def widget_params(self, params=None):
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
        interact(self.widget_plot_params, __manual=True, **intervals)

    def get_params_process(self, process=None, params=None, current=None, fixed=False):
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
        default = DictObj()
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
        return {k:v for k,v in params.items() if k not in self.params_fixed.keys()}

    def optimize(self, fmin=None, vars=None, start=None, max_time=None, *args, **kwargs):
        if fmin is None:
            fmin = sp.optimize.fmin_bfgs
        if vars is None:
            vars = self.sampling_vars
        if start is None:
            start = self.get_params_default()
        if max_time is None:
            callback = None
        else:
            callback = MaxTime(max_time)

        if 'fprime' in signature(fmin).parameters:
            r = fmin(lambda x: nan_to_high(-self.logp_array(x)), self.model.bijection.map(start),
                     fprime=lambda x: np.nan_to_num(-self.dlogp_array(x)), callback=callback, *args, **kwargs)
        else:
            r = fmin(lambda x: nan_to_high(-self.logp_array(x)), self.model.bijection.map(start), callback=callback, *args, **kwargs)
        return self.model.bijection.rmap(r)

    def find_MAP(self, start=None, points=1, plot=False, return_points=False, display=True, powell=True, max_time=None):
        points_list = list()
        if start is None:
            start = self.get_params_current()
        if type(start) is list:
            i = 0
            for s in start:
                i += 1
                points_list.append(('start'+str(i), self.logp_dict(s), s))
        else:
            points_list.append(('start', self.logp_dict(start), start))
        n_starts = len(points_list)
        if self.outputs.get_value() is None:
            print('For find_MAP it is necessary to have observations')
            return start
        if display:
            print('Starting function value (-logp): ' + str(-self.logp_dict(points_list[0][2])))
        if plot:
            plt.figure(0)
            self.plot(params=points_list[0][2], title='start')
            plt.show()
        with self.model:
            i = -1
            points -= 1
            while i < points:
                i += 1
                try:
                    if powell:
                        name, logp, start = points_list[i // 2]
                    else:
                        name, logp, start = points_list[i]
                    if i % 2 == 0 or not powell:#
                        if name.endswith('_bfgs'):
                            if i > n_starts:
                                points += 1
                            continue
                        name += '_bfgs'
                        if display:
                            print('\n' + name)
                        new = self.optimize(fmin=sp.optimize.fmin_bfgs, vars=self.sampling_vars, start=start, max_time=max_time, disp=display)
                    else:
                        if name.endswith('_powell'):
                            if i > n_starts:
                                points += 1
                            continue
                        name += '_powell'
                        if display:
                            print('\n' + name)
                        new = self.optimize(fmin=sp.optimize.fmin_powell, vars=self.sampling_vars, start=start, max_time=max_time, disp=display)
                    points_list.append((name, self.logp_dict(new), new))
                    if plot:
                        plt.figure(i+1)
                        self.plot(params=new, title=name)
                        plt.show()
                except Exception as error:
                    print(error)
                    pass

        optimal = points_list[0]
        for test in points_list:
            if test[1] > optimal[1]:
                optimal = test
        name, logp, params = optimal
        if display:
            #print(params)
            pass
        if return_points is False:
            return params
        else:
            return params, points_list

    def find_MAP_old(self, start=None, points=1, plot=False, return_points=False, display=True, powell=True):
        points_list = list()
        if start is None:
            start = self.get_params_current()
        if type(start) is list:
            i = 0
            for s in start:
                i += 1
                points_list.append(('start'+str(i), self.model.logp(s), s))
        else:
            points_list.append(('start', self.model.logp(start), start))
        n_starts = len(points_list)
        if self.outputs.get_value() is None:
            print('For find_MAP it is necessary to have observations')
            return start
        if display:
            print('Starting function value (-logp): ' + str(-self.model.logp(points_list[0][2])))
        if plot:
            plt.figure(0)
            self.plot(params=points_list[0][2], title='start')
            plt.show()
        with self.model:
            i = -1
            points -= 1
            while i < points:
                i += 1
                try:
                    if powell:
                        name, logp, start = points_list[i // 2]
                    else:
                        name, logp, start = points_list[i]
                    if i % 2 == 0 or not powell:#
                        if name.endswith('_bfgs'):
                            if i > n_starts:
                                points += 1
                            continue
                        name += '_bfgs'
                        if display:
                            print('\n' + name)
                        new = pm.find_MAP(fmin=sp.optimize.fmin_bfgs, vars=self.sampling_vars, start=start, disp=display)
                    else:
                        if name.endswith('_powell'):
                            if i > n_starts:
                                points += 1
                            continue
                        name += '_powell'
                        if display:
                            print('\n' + name)
                        new = pm.find_MAP(fmin=sp.optimize.fmin_powell, vars=self.sampling_vars, start=start, disp=display)
                    points_list.append((name, self.model.logp(new), new))
                    if plot:
                        plt.figure(i+1)
                        self.plot(params=new, title=name)
                        plt.show()
                except:
                    pass

        optimal = points_list[0]
        for test in points_list:
            if test[1] > optimal[1]:
                optimal = test
        name, logp, params = optimal
        if display:
            #print(params)
            pass
        if return_points is False:
            return params
        else:
            return params, points_list

    def sample_hypers(self, start=None, samples=1000, chains=1, trace=None, method='Slice'):
        if start is None:
            start = self.get_params_current()
        if self.outputs.get_value() is None:
            print('For sample_hypers it is necessary to have observations')
            return start
        with self.model:
            if len(self.fixed_vars) > 0:
                step = [ConstantStep(vars=self.fixed_vars)]  # Fue eliminado por error en pymc3
                if len(self.sampling_vars) > 0:
                    if method == 'HMC':
                        step += [pm.HamiltonianMC(vars=self.sampling_vars)]
                    else:
                        step += [RobustSlice(vars=self.sampling_vars)]  # Slice original se cuelga si parte del óptimo
            else:
                if method == 'HMC':

                    step = pm.HamiltonianMC()
                else:
                    step = RobustSlice()
            trace = pm.sample(samples, step=step, start=start, njobs=chains, trace=trace)

        return trace

    def ensemble_hypers(self, start=None, samples=1000, chains=None, ntemps=None):
        if start is None:
            start = self.find_MAP()
        if isinstance(start, dict):
            start = self.dict_to_array(start)
        ndim = len(start)
        if chains is None:
            chains = 2*ndim
        if ntemps is None:
            sampler = emcee.EnsembleSampler(chains, ndim, self.logp_array)
            noise = np.random.normal(loc=1, scale=0.1, size=(chains, ndim))
            p0 = noise * np.ones((chains, 1)) * start
        else:
            sampler = emcee.PTSampler(ntemps, chains, ndim, self.logp_like, self.logp_prior)
            noise = np.random.normal(loc=1, scale=0.1, size=(ntemps, chains, ndim))
            p0 = noise * np.ones((ntemps, chains, 1)) * start

        for result in tqdm(sampler.sample(p0, iterations=samples), total=samples):
            pass

        lnprob, echain = sampler.lnprobability, sampler.chain
        sampler.reset()
        if ntemps is None:
            return lnprob, echain
        else:
            return lnprob[0, :, :], echain[0, :, :]

    def save_model(self, path=None, params=None):
        if path is None:
            path = self.file
        if params is not None:
            self.set_params(params)
        try:
            if os.path.isfile(path):
                os.remove(path)
            with self.model:
                with open(path, 'wb') as f:
                    pickle.dump(self, f, protocol=-1)
            print('Saved model '+path)
        except Exception as details:
            print('Error saving model '+path, details)

    def plot_space(self, space=None, independ=False ,observed=False):
        if space is not None:
            self.set_space(space)
        if independ:
            for i in range(self.space_values.shape[1]):
                plt.figure(i)
                plt.plot(self.space_index, self.space_values[:, i])
        else:
            plt.plot(self.space_index, self.space_values)
        if self.observed_index is not None and observed:
            if independ:
                for i in range(self.space_values.shape[1]):
                    plt.figure(i)
                    plt.plot(self.observed_index, self.inputs_values[:, i], '.k')
            else:
                plt.plot(self.observed_index, self.inputs_values, '.k')

    def plot_hidden(self, big=None):
        if big is None:
            big = config.plot_big
        if big:
            self.plot_hidden_big()
        else:
            self.plot_hidden_normal()

    def plot_hidden_normal(self):
        if self.hidden is not None:
            plt.plot(self.space_index, self.hidden[0:len(self.space_index)],  label='Hidden Processes')

    def plot_hidden_big(self):
        if self.hidden is not None:
            plt.plot(self.space_index, self.hidden[0:len(self.space_index)], linewidth=4, label='Hidden Processes')

    def plot_observations(self, big=None):
        if big is None:
            big = config.plot_big
        if big:
            self.plot_observations_big()
        else:
            self.plot_observations_normal()

    def plot_observations_normal(self):
        if self.outputs_values is not None:
            plt.plot(self.observed_index, self.outputs_values, '.k', ms=10)
            plt.plot(self.observed_index, self.outputs_values, '.r', ms=6, label='Observations')

    def plot_observations_big(self):
        if self.outputs_values is not None:
            plt.plot(self.observed_index, self.outputs_values, '.k', ms=20)
            plt.plot(self.observed_index, self.outputs_values, '.r', ms=15, label='Observations')

    def subprocess(self, subkernel):
        pass

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

