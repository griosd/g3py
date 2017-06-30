import types
import numpy as np
import theano as th
import theano.tensor as tt
from ..libs.tensors import tt_to_num, tt_to_cov, cholesky_robust, makefn
from .hypers import Freedom
from .hypers.means import Mean
from .hypers.kernels import Kernel, KernelSum, WN
from .hypers.mappings import Mapping, Identity
from ..bayesian.models import GraphicalModel, PlotModel
import theano.tensor.slinalg as tsl
from ..libs.plots import show, plot_text
from ..libs import DictObj
import matplotlib.pyplot as plt
from matplotlib import cm
# from ..bayesian.models import TheanoBlackBox


class StochasticProcess(PlotModel):#TheanoBlackBox

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        instance.__init__(*args, **kwargs)
        with instance.active.model:
            instance._check_hypers()
            instance._define_process()
        instance._compile_methods()
        return instance

    def __init__(self, name='SP', space=None, order=None, inputs=None, outputs=None, hidden=None, index=None,
                 distribution=None, active=True, precompile=False, *args, **kwargs):

        ndim = 1
        if space is not None:
            if hasattr(space,'shape'):
                if len(space.shape) > 1:
                    ndim = space.shape[1]
            else:
                ndim = int(space)
        self.nspace = ndim
        self.name = name

        self.th_order = th.shared(np.array([0.0, 1.0], dtype=th.config.floatX), name=self.name + '_order', borrow=False, allow_downcast=True)
        self.th_space = th.shared(np.array([[0.0, 1.0]]*self.nspace, dtype=th.config.floatX).T, name=self.name + '_space', borrow=False, allow_downcast=True)
        self.th_hidden = th.shared(np.array([0.0, 1.0], dtype=th.config.floatX), name=self.name + '_hidden', borrow=False, allow_downcast=True)

        self.th_index = th.shared(np.array([0.0, 1.0], dtype=th.config.floatX), name=self.name + '_index', borrow=False, allow_downcast=True)
        self.th_inputs = th.shared(np.array([[0.0, 1.0]]*self.nspace, dtype=th.config.floatX).T, name=self.name + '_inputs', borrow=False, allow_downcast=True)
        self.th_outputs = th.shared(np.array([0.0, 1.0], dtype=th.config.floatX), name=self.name + '_outputs', borrow=False, allow_downcast=True)

        self.set_space(space=space, hidden=hidden, order=order, inputs=inputs, outputs=outputs, index=index)

        self.distribution = distribution
        if active is True:
            if GraphicalModel.active is None:
                GraphicalModel.active = GraphicalModel('GM_' + self.name)
            self.active = GraphicalModel.active
        elif active is False:
            self.active = GraphicalModel('GM_' + self.name)
        else:
            self.active = active
        self.compiles = DictObj()
        self.precompile = precompile
        super().__init__(*args, **kwargs)

    @property
    def space(self):
        return self.th_space.get_value()
    @space.setter
    def space(self, value):
        self.th_space.set_value(value)

    @property
    def hidden(self):
        return self.th_hidden.get_value()
    @hidden.setter
    def hidden(self, value):
        self.th_hidden.set_value(value)

    @property
    def inputs(self):
        return self.th_inputs.get_value()
    @inputs.setter
    def inputs(self, value):
        self.th_inputs.set_value(value)

    @property
    def outputs(self):
        return self.th_outputs.get_value()
    @outputs.setter
    def outputs(self, value):
        self.th_outputs.set_value(value)

    @property
    def order(self):
        return self.th_order.get_value()
    @order.setter
    def order(self, value):
        self.th_order.set_value(value)

    @property
    def index(self):
        return self.th_index.get_value()
    @index.setter
    def index(self, value):
        self.th_index.set_value(value)

    def default_hypers(self):
        pass

    def _check_hypers(self):
        pass

    def _define_process(self):
        pass

    def _quantiler(self, q=0.975, prior=False, noise=False):
        pass

    def _sampler(self, samples=1, prior=False, noise=False):
        pass

    def _median(self, prior=False, noise=False):
        pass

    def _mean(self, prior=False, noise=False):
        pass

    def _variance(self, prior=False, noise=False):
        pass

    def _covariance(self, prior=False, noise=False):
        pass

    def _predictive(self, prior=False, noise=False):
        pass

    def _logp(self):
        pass

    def _std(self, *args, **kwargs):
        return tt.sqrt(self._variance(*args, **kwargs))

    def _compile_methods(self):
        self.mean = types.MethodType(self._method_name('_mean'), self)
        self.median = types.MethodType(self._method_name('_median'), self)
        self.variance = types.MethodType(self._method_name('_variance'), self)
        self.std = types.MethodType(self._method_name('_std'), self)
        self.covariance = types.MethodType(self._method_name('_covariance'), self)
        self.predictive = types.MethodType(self._method_name('_predictive'), self)
        self.quantiler = types.MethodType(self._method_name('_quantiler'), self)
        self.sampler = types.MethodType(self._method_name('_sampler'), self)
        self.logp = types.MethodType(self._method_name('_logp'), self)

    @staticmethod
    def _method_name(name=None):
        def _method(self, params=None, space=None, hidden=None, inputs=None, outputs=None, prior=False, noise=False, **kwargs):
            if params is None:
                params = self.params_current
            if (space is not None) or (hidden is not None) or (inputs is not None) or (outputs is not None):
                self.set_space(space=space, hidden=hidden, inputs=inputs, outputs=outputs)
            return self._jit_compile(name, prior=prior, noise=noise, **kwargs)(**params)
        return _method

    def _jit_compile(self, method, prior=False, noise=False, **kwargs):
        name = ''
        if prior:
            name += 'prior'
        else:
            name += 'posterior'
        name += method
        if noise:
            name += '_noise'
        if not hasattr(self.compiles, name):
            self.compiles[name] = makefn(self.active.model.vars, getattr(self, method)(prior=prior, noise=noise, **kwargs),
                                         self.precompile)
        return self.compiles[name]

    def set_params(self, params=None):
        if params is not None:
            self.params_current = params

    def set_space(self, space=None, hidden=None, order=None, inputs=None, outputs=None, index=None):
        if space is not None:
            if len(space.shape) < 2:
                space = space.reshape(len(space), 1)
            self.th_space.set_value(space)
        if hidden is not None:
            if len(hidden.shape) > 1:
                hidden = hidden.reshape(len(hidden))
            self.th_hidden.set_value(hidden)
        if order is not None:
            if len(order.shape) > 1:
                order = order.reshape(len(order))
            self.th_order.set_value(order)
        elif self.nspace == 1:
            self.th_order.set_value(self.space.reshape(len(self.space)))

        if inputs is not None:
            if len(inputs.shape) < 2:
                inputs = inputs.reshape(len(inputs), 1)
            self.th_inputs.set_value(inputs)
        if outputs is not None:
            if len(outputs.shape) > 1:
                outputs = outputs.reshape(len(outputs))
            self.th_outputs.set_value(outputs)
        if index is not None:
            if len(index.shape) > 1:
                index = index.reshape(len(index))
            self.th_index.set_value(index)
        elif self.nspace == 1:
            self.th_index.set_value(self.inputs.reshape(len(self.inputs)))
        #check dims
        if len(self.order) != len(self.space):
            self.th_order.set_value(np.arange(len(self.space)))
        if len(self.index) != len(self.inputs):
            self.th_index.set_value(np.arange(len(self.inputs)))

    def observed(self, inputs=None, outputs=None, index=None):
        self.set_space(inputs=inputs, outputs=outputs, index=index)

    def predict(self, params=None, space=None, inputs=None, outputs=None, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False, samples=0, distribution=False, prior=False):
        if params is None:
            params = self.params_current
        self.set_space(space=space, inputs=inputs, outputs=outputs)
        values = DictObj()
        if mean:
            values['mean'] = self.mean(params, prior=prior)
        if var:
            values['variance'] = self.variance(params, prior=prior)
            values['std'] = self.std(params, prior=prior)
        if cov:
            values['covariance'] = self.covariance(params, prior=prior, noise=noise)
        if median:
            values['median'] = self.median(params, prior=prior)
        if quantiles:
            values['quantile_up'] = self.quantiler(params, q=0.975, prior=prior)
            values['quantile_down'] = self.quantiler(params, q=0.025, prior=prior)
        if noise:
            values['noise'] = self.std(params, prior=prior, noise=True)
            values['noise_up'] = self.quantiler(params, q=0.975, prior=prior, noise=True)
            values['noise_down'] = self.quantiler(params, q=0.025, prior=prior, noise=True)
        if samples > 0:
            values['samples'] = self.sampler(params, samples=samples, prior=prior, noise=noise)
        if distribution:
            values['logp'] = lambda x: self.compiles['posterior_logp'](x, space, inputs, outputs, **params)
            values['logpred'] = lambda x: self.compiles['posterior_logpred'](x, space, inputs, outputs, **params)
            values['distribution'] = lambda x: self.compiles['posterior_distribution'](x, space, inputs, outputs, **params)
        return values

    def sample(self, params=None, space=None, inputs=None, outputs=None, samples=1, prior=False):
        S = self.predict(params=params, space=space, inputs=inputs, outputs=outputs, mean=False, var=False, cov=False, median=False, quantiles=False, noise=False, samples=samples, prior=False)
        return S['samples']

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


class EllipticalProcess(StochasticProcess):
    def __init__(self, location: Mean=None, kernel: Kernel=None, degree: Freedom=None, mapping: Mapping=Identity(),
                 noise=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.f_location = location
        self.f_degree = degree
        self.f_mapping = mapping
        if noise:
            self.f_kernel = kernel
            self.f_kernel_noise = KernelSum(self.f_kernel, WN(self.th_space, 'Noise'))
        else:
            self.f_kernel = kernel
            self.f_kernel_noise = self.f_kernel

    def _check_hypers(self):

        self.f_location.check_dims(self.inputs)
        self.f_kernel_noise.check_dims(self.inputs)
        self.f_mapping.check_dims(self.inputs)

        self.f_location.check_hypers(self.name + '_')
        self.f_kernel_noise.check_hypers(self.name + '_')
        self.f_mapping.check_hypers(self.name + '_')

        self.f_location.check_potential()
        self.f_kernel_noise.check_potential()
        self.f_mapping.check_potential()

        if self.f_degree is not None:
            self.f_degree.check_dims(None)
            self.f_degree.check_hypers(self.name + '_')
            self.f_degree.check_potential()

    def default_hypers(self):
        x = np.array(self.inputs_values)
        y = np.squeeze(np.array(self.outputs_values))
        return {**self.f_location.default_hypers_dims(x, y), **self.f_kernel_noise.default_hypers_dims(x, y),
                **self.f_mapping.default_hypers_dims(x, y)}

    def _define_process(self):
        # Basic Tensors
        self.mapping_outputs = tt_to_num(self.f_mapping.inv(self.th_outputs))
        #self.mapping_th = tt_to_num(self.mapping(self.random_th))
        #self.mapping_inv_th = tt_to_num(self.mapping.inv(self.random_th))

        self.prior_location_space = self.f_location(self.th_space)
        self.prior_location_inputs = self.f_location(self.th_inputs)

        self.prior_kernel_space = tt_to_cov(self.f_kernel_noise.cov(self.th_space))
        self.prior_kernel_inputs = tt_to_cov(self.f_kernel_noise.cov(self.th_inputs))
        self.prior_cholesky_space = cholesky_robust(self.prior_kernel_space)

        self.prior_kernel_f_space = self.f_kernel.cov(self.th_space)
        self.prior_kernel_f_inputs = self.f_kernel.cov(self.th_inputs)
        self.prior_cholesky_f_space = cholesky_robust(self.prior_kernel_f_space)

        self.cross_kernel_space_inputs = tt_to_num(self.f_kernel_noise.cov(self.th_space, self.th_inputs))
        self.cross_kernel_f_space_inputs = tt_to_num(self.f_kernel.cov(self.th_space, self.th_inputs))

        self.posterior_location_space = self.prior_location_space + self.cross_kernel_f_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.mapping_outputs - self.prior_location_inputs))

        self.posterior_kernel_space = self.prior_kernel_space - self.cross_kernel_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.cross_kernel_space_inputs.T))
        self.posterior_cholesky_space = cholesky_robust(self.posterior_kernel_space)

        self.posterior_kernel_f_space = self.prior_kernel_f_space - self.cross_kernel_f_space_inputs.dot(
            tsl.solve(self.prior_kernel_inputs, self.cross_kernel_f_space_inputs.T))
        self.posterior_cholesky_f_space = cholesky_robust(self.posterior_kernel_f_space)

    def _location(self, prior=False, noise=False):
        if prior:
            return self.prior_location_space
        else:
            return self.posterior_location_space

    def _kernel(self, prior=False, noise=False):
        if prior:
            if noise:
                return self.prior_kernel_space
            else:
                return self.prior_kernel_f_space
        else:
            if noise:
                return self.posterior_kernel_space
            else:
                return self.posterior_kernel_f_space

    def _cholesky(self, prior=False, noise=False):
        if prior:
            if noise:
                return self.prior_cholesky_space
            else:
                return self.prior_cholesky_f_space
        else:
            if noise:
                return self.posterior_cholesky_space
            else:
                return self.posterior_cholesky_f_space

    def _compile_methods(self):
        super()._compile_methods()
        self.location = types.MethodType(self._method_name('_location'), self)
        self.kernel = types.MethodType(self._method_name('_kernel'), self)
        self.cholesky = types.MethodType(self._method_name('_cholesky'), self)

    def plot_mapping(self, params=None, domain=None, inputs=None, outputs=None, neval=100, title=None, label='mapping'):
        if params is None:
            params = self.get_params_current()
        if outputs is None:
            outputs = self.th_outputs
        if domain is None:
            domain = np.linspace(outputs._mean() - 2 * np.sqrt(outputs.var()),
                                 outputs._mean() + 2 * np.sqrt(outputs.var()), neval)
        #inv_transform = self.compiles['mapping_th'](domain, **params)
        #plt.plot(inv_transform, domain, label='mapping_th')

        #if domain is None:
        #    domain = np.linspace(outputs.mean() - 2*np.sqrt(outputs.var()), outputs.mean() + 2*np.sqrt(outputs.var()), neval)
        transform = self.compiles['mapping_inv_th'](domain, **params)
        plt.plot(domain, transform, label=label)

        if title is None:
            title = 'Mapping'
        plot_text(title, 'Domain y', 'Domain T(y)')

    def plot_kernel(self, params=None, space=None, inputs=None, centers=[1/10, 1/2, 9/10]):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.th_space
        if inputs is None:
            inputs = self.inputs_values
        ksi = self.compiles['kernel_space_inputs'](space, inputs, **params).T
        for ind in centers:
            plt.plot(self.th_order, ksi[int(len(ksi)*ind), :], label='k(x_'+str(int(len(ksi)*ind))+')')
        plot_text('Kernel', 'Space x', 'Kernel value k(x,v)')

    def plot_kernel2D(self):
        pass

    def plot_concentration(self, params=None, space=None, color=True, figsize=(6, 6)):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.th_space
        concentration_matrix = self.compiles['kernel_space'](space, **params)
        if color:
            plt.figure(None, figsize)
            v = np.max(np.abs(concentration_matrix))
            plt.imshow(concentration_matrix, cmap=cm.seismic, vmax=v, vmin=-v)
        else:
            plt.matshow(concentration_matrix)
        plot_text('Concentration', 'Space x', 'Space x', legend=False)

    def plot_location(self, params=None, space=None):
        if params is None:
            params = self.get_params_current()
        if space is None:
            space = self.th_space
        plt.plot(self.th_order, self.compiles['location_space'](space, **params), label='location')
        plot_text('Location', 'Space x', 'Location value m(x)')

    def plot_model(self, params=None, indexs=None, kernel=True, mapping=True, marginals=True, bivariate=True):
        if indexs is None:
            indexs = [self.th_index[len(self.th_index)//2], self.th_index[len(self.th_index)//2]+1]
        if kernel:
            plt.subplot(121)
            self.plot_kernel(params=params)
        if mapping:
            plt.subplot(122)
            self.plot_mapping(params=params)
        show()

        if marginals:
            plt.subplot(121)
            self.plot_distribution(index=indexs[0], params=params, space=self.th_space[indexs[0]:indexs[0]+1, :], prior=True)
            self.plot_distribution(index=indexs[0], params=params, space=self.th_space[indexs[0]:indexs[0]+1, :])
            plt.subplot(122)
            self.plot_distribution(index=indexs[1], params=params, space=self.th_space[indexs[1]:indexs[1]+1, :], prior=True)
            self.plot_distribution(index=indexs[1], params=params, space=self.th_space[indexs[1]:indexs[1]+1, :])
            show()
        if bivariate:
            self.plot_distribution2D(indexs=indexs, params=params, space=self.th_space[indexs, :])
            show()


class CopulaProcess(StochasticProcess):
    def __init__(self, copula: StochasticProcess=None, marginal: Mapping=None):
        pass
