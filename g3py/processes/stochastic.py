import types
import numpy as np
import theano as th
import theano.tensor as tt
import theano.tensor.nlinalg as tnl
#import pymc3 as pm
from ..libs.tensors import tt_to_num, tt_to_cov, tt_to_bounded, cholesky_robust, makefn, gradient
from .hypers import Freedom
from .hypers.means import Mean
from .hypers.kernels import Kernel, KernelSum, WN
from .hypers.mappings import Mapping, Identity
from ..bayesian.models import GraphicalModel, PlotModel
import theano.tensor.slinalg as tsl
from ..bayesian.selection import optimize
from ..bayesian.average import mcmc_ensemble, chains_to_datatrace
from ..libs.plots import show, plot_text
from ..libs import DictObj, clone
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit
# from ..bayesian.models import TheanoBlackBox


class StochasticProcess(PlotModel):#TheanoBlackBox

    def __init__(self, space=None, order=None, inputs=None, outputs=None, hidden=None, index=None,
                 name='SP', distribution=None, active=False, precompile=False, *args, **kwargs):
        #print('StochasticProcess__init_')
        ndim = 1
        self.makefn = makefn
        if space is not None:
            if hasattr(space, 'shape'):
                if len(space.shape) > 1:
                    ndim = space.shape[1]
            else:
                ndim = int(space)
        self.nspace = ndim
        self.name = name

        self.th_order = th.shared(np.array([0.0, 1.0, 2.0], dtype=th.config.floatX),
                                  name=self.name + '_order', borrow=False, allow_downcast=True)
        self.th_space = th.shared(np.array([[0.0, 1.0, 2.0]]*self.nspace, dtype=th.config.floatX).T,
                                  name=self.name + '_space', borrow=False, allow_downcast=True)

        self.th_index = th.shared(np.array([0.0, 1.0], dtype=th.config.floatX),
                                  name=self.name + '_index', borrow=False, allow_downcast=True)
        self.th_inputs = th.shared(np.array([[0.0, 1.0]]*self.nspace, dtype=th.config.floatX).T,
                                   name=self.name + '_inputs', borrow=False, allow_downcast=True)
        self.th_outputs = th.shared(np.array([0.0, 1.0], dtype=th.config.floatX),
                                    name=self.name + '_outputs', borrow=False, allow_downcast=True)
        self.is_observed = False
        self.np_hidden = None

        self.th_space_ = tt.matrix(self.name + '_space_th', dtype=th.config.floatX)
        self.th_inputs_ = tt.matrix(self.name + '_inputs_th', dtype=th.config.floatX)
        self.th_outputs_ = tt.vector(self.name + '_outputs_th', dtype=th.config.floatX)
        self.th_space_.tag.test_value = np.array([[0.0, 1.0, 2.0]]*self.nspace, dtype=th.config.floatX).T
        self.th_inputs_.tag.test_value = np.array([[0.0, 1.0]]*self.nspace, dtype=th.config.floatX).T
        self.th_outputs_.tag.test_value = np.array([0.0, 1.0], dtype=th.config.floatX)
        self.th_scalar = tt.scalar('q', dtype=th.config.floatX)
        self.th_scalar.tag.test_value = np.float32(1)
        self.th_vector = tt.vector(self.name + '_vector_th', dtype=th.config.floatX)
        self.th_vector.tag.test_value = np.array([0.0, 1.0], dtype=th.config.floatX)
        self.th_matrix = tt.matrix(self.name + '_matrix_th', dtype=th.config.floatX)
        self.th_matrix.tag.test_value = np.array([[0.0, 1.0]]*self.nspace, dtype=th.config.floatX).T

        self.distribution = distribution
        if active is True:
            if GraphicalModel.active is None:
                GraphicalModel.active = GraphicalModel('GM_' + self.name)
            self.active = GraphicalModel.active
        elif active is False:
            self.active = GraphicalModel('GM_' + self.name)
        else:
            self.active = active
        self.active.add_component(self)
        self.compiles = DictObj()
        self.compiles_trans = DictObj()
        self.precompile = precompile
        super().__init__(*args, **kwargs)
        #print('_define_process')
        with self.model:
            self._check_hypers()
            self.th_define_process()
        #print('set_space')
        self.set_space(space=space, hidden=hidden, order=order, inputs=inputs, outputs=outputs, index=index)
        #print('_compile_methods')
        self._compile_methods()
        #print('StochasticProcess__end_')

    def set_params(self, *args, **kwargs):
        return self.active.set_params(*args, **kwargs)

    def set_space(self, space=None, hidden=None, order=None, inputs=None, outputs=None, index=None):
        if space is not None:
            if len(space.shape) < 2:
                space = space.reshape(len(space), 1)
            self.space = space
        if hidden is not None:
            if len(hidden.shape) > 1:
                hidden = hidden.reshape(len(hidden))
            self.hidden = hidden
        if order is not None:
            if len(order.shape) > 1:
                order = order.reshape(len(order))
            self.order = order
        elif self.nspace == 1:
            self.order = self.space.reshape(len(self.space))

        if inputs is not None:
            if len(inputs.shape) < 2:
                inputs = inputs.reshape(len(inputs), 1)
            self.inputs = inputs
        if outputs is not None:
            if len(outputs.shape) > 1:
                outputs = outputs.reshape(len(outputs))
            self.outputs = outputs
        if index is not None:
            if len(index.shape) > 1:
                index = index.reshape(len(index))
            self.index = index
        elif self.nspace == 1:
            self.index = self.inputs.reshape(len(self.inputs))
        #check dims
        if len(self.order) != len(self.space):
            self.order = np.arange(len(self.space))
        if len(self.index) != len(self.inputs):
            self.index = np.arange(len(self.inputs))

    def observed(self, inputs=None, outputs=None, order=None, index=None, hidden=None):
        self.set_space(inputs=inputs, outputs=outputs, order=order, index=index, hidden=hidden)
        if inputs is None and outputs is None and inputs is None:
            self.is_observed = False
        else:
            self.is_observed = True

    @property
    def model(self):
        return self.active.model

    @property
    def params(self):
        return self.active.params

    @property
    def space(self):
        return self.th_space.get_value(borrow=False)
    @space.setter
    def space(self, value):
        self.th_space.set_value(value, borrow=False)

    @property
    def hidden(self):
        return self.np_hidden
    @hidden.setter
    def hidden(self, value):
        self.np_hidden = value

    @property
    def inputs(self):
        return self.th_inputs.get_value(borrow=False)
    @inputs.setter
    def inputs(self, value):
        self.th_inputs.set_value(value, borrow=False)

    @property
    def outputs(self):
        return self.th_outputs.get_value(borrow=False)
    @outputs.setter
    def outputs(self, value):
        self.th_outputs.set_value(value, borrow=False)

    @property
    def order(self):
        return self.th_order.get_value(borrow=False)
    @order.setter
    def order(self, value):
        self.th_order.set_value(value, borrow=False)

    @property
    def index(self):
        return self.th_index.get_value(borrow=False)
    @index.setter
    def index(self, value):
        self.th_index.set_value(value, borrow=False)

    def default_hypers(self):
        pass

    def _check_hypers(self):
        pass

    def th_define_process(self):
        pass

    def quantiler(self, q=0.975, prior=False, noise=False):
        pass

    def sampler(self, samples=1, prior=False, noise=False):
        pass

    def th_median(self, prior=False, noise=False):
        pass

    def th_mean(self, prior=False, noise=False):
        pass

    def th_variance(self, prior=False, noise=False):
        pass

    def th_covariance(self, prior=False, noise=False):
        pass

    def th_predictive(self, prior=False, noise=False):
        pass

    def th_std(self, *args, **kwargs):
        return tt.sqrt(self.th_variance(*args, **kwargs))

    def th_logp(self, prior=False, noise=False):
        if prior:
            random_vars = self.model.free_RVs
        else:
            random_vars = self.model.basic_RVs
        factors = [var.logpt for var in random_vars] + self.model.potentials
        return tt.add(*map(tt.sum, factors))

    def th_dlogp(self, dvars=None, *args, **kwargs):
        return tt_to_num(gradient(self.th_logp(*args, **kwargs), dvars))

    def th_loglike(self, prior=False, noise=False):
        factors = [var.logpt for var in self.model.observed_RVs]
        return tt.add(*map(tt.sum, factors))

    def _compile_methods(self):
        self.mean = types.MethodType(self._method_name('th_mean'), self)
        self.median = types.MethodType(self._method_name('th_median'), self)
        self.variance = types.MethodType(self._method_name('_variance'), self)
        self.std = types.MethodType(self._method_name('th_std'), self)
        self.covariance = types.MethodType(self._method_name('th_covariance'), self)
        self.predictive = types.MethodType(self._method_name('th_predictive'), self)

        #self.quantiler = types.MethodType(self._method_name('_quantiler'), self)
        #self.sampler = types.MethodType(self._method_name('_sampler'), self)

        self.logp = types.MethodType(self._method_name('th_logp'), self)
        self.dlogp = types.MethodType(self._method_name('th_dlogp'), self)
        self.loglike = types.MethodType(self._method_name('th_loglike'), self)

        _ = self.logp(array=True), self.dlogp(array=True)

    def _method_name(self, method=None):
        def _method(self, params=None, space=None, inputs=None, outputs=None, prior=False, noise=False, array=False, *args, **kwargs):
            if params is None:
                if array:
                    params = self.active.dict_to_array(self.params)
                else:
                    params = self.params
            if space is None:
                space = self.space
            if inputs is None:
                inputs = self.inputs
            if outputs is None:
                outputs = self.outputs
            #return self._jit_compile(method, prior=prior, noise=noise, array=array, *args, **kwargs)(self.space, self.inputs, self.outputs, params)
            name = ''
            if prior:
                name += 'prior'
            else:
                name += 'posterior'
            name += method.replace('th', '') # delete th
            if noise:
                name += '_noise'
            if len(args) > 0:
                name += str(args)
            if len(kwargs) > 0:
                name += str(kwargs)
            if not hasattr(self.compiles, name):
                th_vars = [self.th_space_, self.th_inputs_, self.th_outputs_] + self.model.vars
                self.compiles[name] = self.makefn(th_vars, getattr(self, method)(prior=prior, noise=noise, *args, **kwargs),
                                             givens = [(self.th_space, self.th_space_), (self.th_inputs, self.th_inputs_), (self.th_outputs, self.th_outputs_)],
                                             bijection=None, precompile=self.precompile)
            if array:
                if not hasattr(self.compiles, 'array_' + name):
                    self.compiles['array_' + name] = self.compiles[name].clone(self.active.bijection.rmap)
                name = 'array_' + name
            return self.compiles[name](params, space, inputs, outputs)
        return _method

    @property
    def executed(self):
        return {k: v.executed for k, v in self.compiles.items()}

    def predict(self, params=None, space=None, inputs=None, outputs=None, mean=True, std=True, var=False, cov=False, median=False,
                quantiles=False, noise=False, samples=0, distribution=False, prior=False):
        if params is None:
            params = self.params
        if not self.is_observed:
            prior = True
        if space is None:
            space = self.space
        if inputs is None:
            inputs = self.inputs
        if outputs is None:
            outputs = self.outputs
        values = DictObj()
        if mean:
            values['mean'] = self.mean(params, space, inputs, outputs, prior=prior)
        if var:
            values['variance'] = self.variance(params, space, inputs, outputs, prior=prior)
        if std:
            values['std'] = self.std(params, space, inputs, outputs, prior=prior)
        if cov:
            values['covariance'] = self.covariance(params, space, inputs, outputs, prior=prior, noise=noise)
        if median:
            values['median'] = self.median(params, space, inputs, outputs, prior=prior)
        if quantiles:
            values['quantile_up'] = self.quantiler(params, space, inputs, outputs, q=0.975, prior=prior)
            values['quantile_down'] = self.quantiler(params, space, inputs, outputs, q=0.025, prior=prior)
        if noise:
            values['noise'] = self.std(params, space, inputs, outputs, prior=prior, noise=True)
            values['noise_up'] = self.quantiler(params, space, inputs, outputs, q=0.975, prior=prior, noise=True)
            values['noise_down'] = self.quantiler(params, space, inputs, outputs, q=0.025, prior=prior, noise=True)
        if samples > 0:
            values['samples'] = self.sampler(params, space, inputs, outputs, samples=samples, prior=prior, noise=False)
        if distribution:
            values['logp'] = lambda x: self.compiles['posterior_logp'](x, space, inputs, outputs, **params)
            values['logpred'] = lambda x: self.compiles['posterior_logpred'](x, space, inputs, outputs, **params)
            values['distribution'] = lambda x: self.compiles['posterior_distribution'](x, space, inputs, outputs, **params)
        return values

    def find_MAP(self, start=None, points=1, plot=False, return_points=False, display=True,
                 powell=True, max_time=None):
        logp = lambda p: self.compiles.array_posterior_logp(p, self.space, self.inputs, self.outputs)
        dlogp = lambda p: self.compiles.array_posterior_dlogp(p, self.space, self.inputs, self.outputs)

        points_list = list()
        if start is None:
            start = self.params
        # if process._fixed_chain is None:
        #    logp = process.logp_array
        # else:
        #    logp = process.logp_fixed_chain

        if type(start) is list:
            i = 0
            for s in start:
                i += 1
                points_list.append(('start' + str(i), logp(self.active.dict_to_array(s)), s))
        else:
            points_list.append(('start', logp(self.active.dict_to_array(start)), start))
        n_starts = len(points_list)
        if self.outputs is None:  # .get_value()
            print('For find_MAP it is necessary to have observations')
            return start
        if display:
            print('Starting function value (-logp): ' + str(-logp(self.active.dict_to_array(points_list[0][2]))))
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
                        name, _, start = points_list[i // 2]
                    else:
                        name, _, start = points_list[i]
                    if i % 2 == 0 or not powell:  #
                        if name.endswith('_bfgs'):
                            if i > n_starts:
                                points += 1
                            continue
                        name += '_bfgs'
                        if display:
                            print('\n' + name)
                        new = optimize(logp=logp, start=self.active.dict_to_array(start), dlogp=dlogp, fmin='bfgs',
                                       max_time=max_time, disp=display)
                    else:
                        if name.endswith('_powell'):
                            if i > n_starts:
                                points += 1
                            continue
                        name += '_powell'
                        if display:
                            print('\n' + name)
                        new = optimize(logp=logp, start=self.active.dict_to_array(start), fmin='powell', max_time=max_time,
                                       disp=display)
                    points_list.append((name, logp(new), self.active.array_to_dict(new)))
                    if plot:
                        plt.figure(i + 1)
                        self.plot(params=new, title=name)
                        plt.show()
                except Exception as error:
                    print(error)
                    pass

        optimal = points_list[0]
        for test in points_list:
            if test[1] > optimal[1]:
                optimal = test
        _, _, params = optimal
        params = DictObj(params)
        if display:
            # print(params)
            pass
        if return_points is False:
            return params
        else:
            return params, points_list

    def sample_hypers(self, start=None, samples=1000, chains=None, ntemps=None, raw=False, noise_mult=0.1, noise_sum=0.01,
                      burnin_tol=0.001, burnin_method='multi-sum', outlayer_percentile=0.0005):
        if start is None:
            start = self.find_MAP()
        if isinstance(start, dict):
            start = self.active.dict_to_array(start)

        if self.active.sampling_dims is None:
            self.active.calc_dimensions()
        ndim = len(self.active.sampling_dims)

        if len(start.shape) == 1:
            start = start[self.active.sampling_dims]
        elif len(start.shape) == 2:
            start = start[:, self.active.sampling_dims]
        elif len(start.shape) == 3:
            start = start[:, 0, self.active.sampling_dims]

        if ntemps is None:
            logp =  lambda p: self.compiles.array_posterior_logp(p, self.space, self.inputs, self.outputs)
            loglike = None
            logprior = None
        else:
            logp = None
            loglike = lambda p: self.compiles.array_posterior_loglike(p, self.space, self.inputs, self.outputs)
            logprior = lambda p: self.compiles.array_prior_logp(p, self.space, self.inputs, self.outputs)

        lnprob, echain = mcmc_ensemble(ndim, samples=samples, chains=chains, ntemps=ntemps, start=start,
                                       logp=logp, loglike=loglike, logprior=logprior,
                                       noise_mult=noise_mult, noise_sum=noise_sum)

        complete_chain = np.empty((echain.shape[0], echain.shape[1], self.active.ndim))
        complete_chain[:, :, self.active.sampling_dims] = echain
        if len(self.active.fixed_dims)>0:
            complete_chain[:, :, self.active.fixed_dims] = self.active._fixed_array[self.active.fixed_dims]
        if raw:
            return complete_chain, lnprob
        else:

            return chains_to_datatrace(self, complete_chain, ll=lnprob, burnin_tol=burnin_tol,
                                       burnin_method=burnin_method, burnin_dims=self.active.sampling_dims,
                                       outlayer_percentile=outlayer_percentile)

    @property
    def ndim(self):
        return self.active.ndim

    def logp_chain(self, chain):
        out = np.empty(len(chain))
        for i in range(len(out)):
            out[i] = self.logp(chain[i], array=True)
        return out


class EllipticalProcess(StochasticProcess):
    def __init__(self, location: Mean=None, kernel: Kernel=None, degree: Freedom=None, mapping: Mapping=Identity(),
                 noise=True, *args, **kwargs):
        #print('EllipticalProcess__init__')

        self.f_location = location
        self.f_degree = degree
        self.f_mapping = mapping
        if noise:
            self.f_kernel = kernel
            self.f_kernel_noise = KernelSum(self.f_kernel, WN(name='Noise')) #self.th_space,
        else:
            self.f_kernel = kernel
            self.f_kernel_noise = self.f_kernel

        super().__init__(*args, **kwargs)

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
        x = self.inputs
        y = self.outputs
        return {**self.f_location.default_hypers_dims(x, y), **self.f_kernel_noise.default_hypers_dims(x, y),
                **self.f_mapping.default_hypers_dims(x, y)}

    def th_define_process(self):
        #print('stochastic_define_process')
        # Basic Tensors
        self.mapping_outputs = tt_to_num(self.f_mapping.inv(self.th_outputs))
        #self.mapping_th = tt_to_num(self.f_mapping(self.random_th))
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

    def th_mapping(self, prior=False, noise=False):
        return self.mapping_outputs

    def th_location(self, prior=False, noise=False):
        if prior:
            return self.prior_location_space
        else:
            return self.posterior_location_space

    def th_kernel(self, prior=False, noise=False):
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

    def th_cholesky(self, prior=False, noise=False):
        return cholesky_robust(self.th_kernel(prior=prior, noise=noise))

    def th_kernel_diag(self, prior=False, noise=False):
        return tt_to_bounded(tnl.extract_diag(self.th_kernel(prior=prior, noise=noise)), 0)

    def th_kernel_sd(self, prior=False, noise=False):
        return tt.sqrt(self.th_kernel_diag(prior=prior, noise=noise))

    def _compile_methods(self):
        super()._compile_methods()
        self.mapping = types.MethodType(self._method_name('th_mapping'), self)
        self.location = types.MethodType(self._method_name('th_location'), self)
        self.kernel = types.MethodType(self._method_name('th_kernel'), self)
        self.cholesky = types.MethodType(self._method_name('th_cholesky'), self)
        self.kernel_diag = types.MethodType(self._method_name('th_kernel_diag'), self)
        self.kernel_sd = types.MethodType(self._method_name('th_kernel_sd'), self)

    def plot_mapping(self, params=None, domain=None, inputs=None, outputs=None, neval=100, title=None, label='mapping'):
        if params is None:
            params = self.get_params_current()
        if outputs is None:
            outputs = self.outputs
        if domain is None:
            mean = np.mean(outputs)
            std = np.std(outputs)
            domain = np.linspace(mean - 2 * std, mean + 2 * std, neval)
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
            params = self.params
        if space is None:
            space = self.th_space
        if inputs is None:
            inputs = self.inputs_values
        ksi = self.compiles['kernel_space_inputs'](space, inputs, **params).T
        for ind in centers:
            plt.plot(self.order, ksi[int(len(ksi)*ind), :], label='k(x_'+str(int(len(ksi)*ind))+')')
        plot_text('Kernel', 'Space x', 'Kernel value k(x,v)')

    def plot_kernel2D(self):
        pass

    def plot_concentration(self, params=None, space=None, color=True, figsize=(6, 6)):
        if params is None:
            params = self.params
        if space is None:
            space = self.th_space
        concentration_matrix = self.compiles.kernel_space(space, **params)
        if color:
            plt.figure(None, figsize)
            v = np.max(np.abs(concentration_matrix))
            plt.imshow(concentration_matrix, cmap=cm.seismic, vmax=v, vmin=-v)
        else:
            plt.matshow(concentration_matrix)
        plot_text('Concentration', 'Space x', 'Space x', legend=False)

    def plot_location(self, params=None, space=None):
        if params is None:
            params = self.params
        if space is None:
            space = self.th_space
        plt.plot(self.order, self.compiles.location_space(space, **params), label='location')
        plot_text('Location', 'Space x', 'Location value m(x)')

    def plot_model(self, params=None, indexs=None, kernel=True, mapping=True, marginals=True, bivariate=True):
        if indexs is None:
            indexs = [self.index[len(self.index)//2], self.index[len(self.index)//2]+1]
        if kernel:
            plt.subplot(121)
            self.plot_kernel(params=params)
        if mapping:
            plt.subplot(122)
            self.plot_mapping(params=params)
        show()

        if marginals:
            plt.subplot(121)
            self.plot_distribution(index=indexs[0], params=params, space=self.space[indexs[0]:indexs[0]+1, :], prior=True)
            self.plot_distribution(index=indexs[0], params=params, space=self.space[indexs[0]:indexs[0]+1, :])
            plt.subplot(122)
            self.plot_distribution(index=indexs[1], params=params, space=self.space[indexs[1]:indexs[1]+1, :], prior=True)
            self.plot_distribution(index=indexs[1], params=params, space=self.space[indexs[1]:indexs[1]+1, :])
            show()
        if bivariate:
            self.plot_distribution2D(indexs=indexs, params=params, space=self.space[indexs, :])
            show()


class CopulaProcess(StochasticProcess):
    def __init__(self, copula: StochasticProcess=None, marginal: Mapping=None):
        pass
