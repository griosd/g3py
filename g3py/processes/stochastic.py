import os
import types

import matplotlib.pyplot as plt
import numpy as np
import theano as th
import theano.tensor as tt

from ..bayesian.average import mcmc_ensemble, chains_to_datatrace, plot_datatrace
from ..bayesian.models import GraphicalModel, PlotModel
from ..bayesian.selection import optimize
from ..libs import DictObj, save_pkl, load_pkl, load_datatrace, save_datatrace
from ..libs.tensors import tt_to_num, makefn, gradient
from multiprocessing import Pool
# from ..bayesian.models import TheanoBlackBox

zero32 = np.float32(0.0)


class StochasticProcess(PlotModel):#TheanoBlackBox

    def __init__(self, space=None, order=None, inputs=None, outputs=None, hidden=None, index=None,
                 name='SP', distribution=None, active=False, precompile=False, file=None, load=True, compile_logp=True,
                 *args, **kwargs):
        if file is not None and load:
            try:
                load = load_pkl(file)
                self.__dict__.update(load.__dict__)
                self._compile_methods(compile_logp)
                print('Loaded model ' + file)
                self.set_space(space=space, hidden=hidden, order=order, inputs=inputs, outputs=outputs, index=index)
                return
            except:
                print('Model Not Found in '+str(file))
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

        self.th_order = th.shared(np.array([0.0, 1.0], dtype=th.config.floatX),
                                  name=self.name + '_order', borrow=False, allow_downcast=True)
        self.th_space = th.shared(np.array([[0.0, 1.0]]*self.nspace, dtype=th.config.floatX).T,
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
        self.th_space_.tag.test_value = np.array([[0.0, 1.0]]*self.nspace, dtype=th.config.floatX).T
        self.th_inputs_.tag.test_value = np.array([[0.0, 1.0]]*self.nspace, dtype=th.config.floatX).T
        self.th_outputs_.tag.test_value = np.array([0.0, 1.0], dtype=th.config.floatX)
        self.th_scalar = tt.scalar(self.name + '_scalar_th', dtype=th.config.floatX)
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

        self.precompile = precompile
        super().__init__(*args, **kwargs)
        #print('_define_process')
        with self.model:
            self._check_hypers()
            self.th_define_process()
            self.active.compile_components()
        #print('set_space')
        self.set_space(space=space, hidden=hidden, order=order, inputs=inputs, outputs=outputs, index=index)
        #print('_compile_methods')
        self._compile_methods(compile_logp)
        if hidden is None:
            self.hidden = hidden

        #print('StochasticProcess__end_')
        if file is not None:
            self.file = file
            try:
                self.save()
            except:
                print('Error in file '+str(file))

    def save(self, path=None, params=None):
        if path is None:
            path = self.file
        if params is not None:
            self.set_params(params)
        try:
            if os.path.isfile(path):
                os.remove(path)
            with self.model:
                save_pkl(self, path)
            print('Model saved on '+path)
        except Exception as details:
            print('Error saving model '+path, details)

    def set_params(self, *args, **kwargs):
        return self.active.set_params(*args, **kwargs)

    def params_random(self, *args, **kwargs):
        """
        Alias for the method .active.params_random()
        """
        return self.active.params_random(*args, **kwargs)

    def params_datatrace(self, *args, **kwargs):
        return self.active.params_datatrace(*args, **kwargs)

    def transform_params(self, *args, **kwargs):
        return self.active.transform_params(*args, **kwargs)

    def params_process(self, process=None, params=None, current=None, fixed=False):
        if process is None:
            process = self
        if params is None:
            params = process.params
        if current is None:
            current = self.params
        params_transform = {k.replace(process.name, self.name, 1): v for k, v in params.items()}
        params_return = DictObj({k: v for k, v in params_transform.items() if k in current.keys()})
        params_return.update({k: v for k, v in current.items() if k not in params_transform.keys()})
        if fixed:
            params_return.update(self.params_fixed)
        return params_return

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
        """
        This function asign the observations to the gp and calculates the default parameters
        Args:
            inputs (numpy.ndarray): the inputs of the process
            outputs (numpy.ndarray): the outputs (observations) of the process
            order (numpy.ndarray): For multidimensional process, the order indicates the order in
                which the domain (space) is plotted.:
            index (numpy.ndarray): It is the index of the observations
            hidden (numpy.ndarray): The set of values from where the observations are taken
        """
        self.set_space(inputs=inputs, outputs=outputs, order=order, index=index, hidden=hidden)
        if inputs is None and outputs is None:
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
    def params_default(self):
        return self.active.params_default

    @property
    def params_test(self):
        return self.active.params_test

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

    def sampler(self, samples=1, prior=False, noise=False):
        pass

    def quantiler(self, q=0.975, prior=False, noise=False, simulations=None):
        pass

    def th_median(self, prior=False, noise=False, simulations=None):
        pass

    def th_mean(self, prior=False, noise=False, simulations=None):
        pass

    def th_variance(self, prior=False, noise=False, simulations=None):
        pass

    def th_covariance(self, prior=False, noise=False):
        pass

    def th_logpredictive(self, prior=False, noise=False):
        pass

    def th_cross_mean(self, prior=False, noise=False, cross_kernel=None):
        pass

    def th_std(self, *args, **kwargs):
        if self.th_variance(*args, **kwargs) is not None:
            return tt.sqrt(self.th_variance(*args, **kwargs))
        else:
            return None

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

    def th_error_l1(self, prior=False, noise=False):
        mean = self.th_mean(prior=prior, noise=noise)
        if mean is not None:
            return tt.mean(tt.abs_(self.th_vector - mean))

    def th_error_l2(self, prior=False, noise=False):
        mean = self.th_mean(prior=prior, noise=noise)
        if mean is not None:
            return tt.mean(tt.pow(self.th_vector - mean, 2))

    def th_error_mse(self, prior=False, noise=False):
        return tt.mean(tt.abs_(self.th_vector - self.th_outputs))**2 + tt.var(tt.abs_(self.th_vector - self.th_outputs))

    def _compile_methods(self, compile_logp=True):
        reset_space = self.space
        reset_hidden = self.hidden
        reset_order = self.order
        reset_inputs = self.inputs
        reset_outputs = self.outputs
        reset_index = self.index
        reset_observed = self.is_observed

        self.set_space(space=self.th_space_.tag.test_value, hidden=self.th_vector.tag.test_value,
                       inputs=self.th_inputs_.tag.test_value, outputs=self.th_outputs_.tag.test_value)
        if self.compiles is None:
            self.compiles = DictObj()
        if self.th_mean() is not None:
            self.mean = types.MethodType(self._method_name('th_mean'), self)
        if self.th_median() is not None:
            self.median = types.MethodType(self._method_name('th_median'), self)
        if self.th_variance() is not None:
            self.variance = types.MethodType(self._method_name('th_variance'), self)
        if self.th_std() is not None:
            self.std = types.MethodType(self._method_name('th_std'), self)
        if self.th_covariance() is not None:
            self.covariance = types.MethodType(self._method_name('th_covariance'), self)
        if self.th_logpredictive() is not None:
            self.logpredictive = types.MethodType(self._method_name('th_logpredictive'), self)
        if self.th_error_l1() is not None:
            self.error_l1 = types.MethodType(self._method_name('th_error_l1'), self)
        if self.th_error_l2() is not None:
            self.error_l2 = types.MethodType(self._method_name('th_error_l2'), self)
        if self.th_error_mse() is not None:
            self.error_mse = types.MethodType(self._method_name('th_error_mse'), self)

        # self.density = types.MethodType(self._method_name('th_density'), self)

        #self.quantiler = types.MethodType(self._method_name('_quantiler'), self)
        #self.sampler = types.MethodType(self._method_name('_sampler'), self)

        self.logp = types.MethodType(self._method_name('th_logp'), self)
        self.dlogp = types.MethodType(self._method_name('th_dlogp'), self)
        self.loglike = types.MethodType(self._method_name('th_loglike'), self)

        self.is_observed = True
        if compile_logp:
            _ = self.logp(array=True)
            _ = self.logp(array=True, prior=True)
            # _ = self.loglike(array=True)
            try:
                _ = self.dlogp(array=True)
            except Exception as m:
                print('Compiling dlogp error:', m)
        self.is_observed = reset_observed
        self.set_space(space=reset_space, hidden=reset_hidden, order=reset_order,
                       inputs=reset_inputs, outputs=reset_outputs, index=reset_index)

    def lambda_method(self, *args, **kwargs):
        pass

    @staticmethod
    def _method_name(method=None):
        def lambda_method(self, params=None, space=None, inputs=None, outputs=None, vector=[], prior=False, noise=False, array=False, *args, **kwargs):
            if params is None:
                if array:
                    params = self.active.dict_to_array(self.params)
                else:
                    params = self.params
            elif not array:
                params = self.filter_params(params)
            if inputs is None and not self.is_observed:
                prior = True
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
                #print(method)
                #if method in ['th_logpredictive', 'th_error_l1', 'th_error_l2']:
                #    th_vars = [self.th_space_, self.th_inputs_, self.th_outputs_, self.th_vector] + self.model.vars
                #else:
                th_vars = [self.th_space_, self.th_inputs_, self.th_outputs_, self.th_vector] + self.model.vars
                self.compiles[name] = self.makefn(th_vars, getattr(self, method)(prior=prior, noise=noise, *args, **kwargs),
                                             givens = [(self.th_space, self.th_space_), (self.th_inputs, self.th_inputs_), (self.th_outputs, self.th_outputs_)],
                                             bijection=None, precompile=self.precompile)
            if array:
                if not hasattr(self.compiles, 'array_' + name):
                    self.compiles['array_' + name] = self.compiles[name].clone(self.active.bijection.rmap)
                name = 'array_' + name
            return self.compiles[name](params, space, inputs, outputs, vector)
        return lambda_method

    @property
    def executed(self):
        return {k: v.executed for k, v in self.compiles.items()}

    @property
    def transformations(self):
        return self.active.transformations

    @property
    def potentials(self):
        return self.active.potentials

    def predict(self, params=None, space=None, inputs=None, outputs=None, mean=True, std=True, var=False, cov=False,
                median=False, quantiles=False, quantiles_noise=False, samples=0, distribution=False,
                prior=False, noise=False, simulations=None):
        """
        Predict a stochastic process with each feature of the process.
        Args:
            params (g3py.libs.DictObj): Contains the hyperparameters of the stochastic process
            space (numpy.ndarray): the domain space of the process
            inputs (numpy.ndarray): the inputs of the process
            outputs (numpy.ndarray): the outputs (observations) of the process
            mean (bool): Determines whether the mean is displayed
            std (bool): Determines whether the standard deviation is displayed
            var (bool): Determines whether the variance is displayed
            cov (bool): Determines whether the covariance is displayed
            median (bool): Determines whether the median is displayed
            quantiles (bool): Determines whether the quantiles (95% of confidence) are displayed
            quantiles_noise (bool): Determines whether the noise is considered for calculating (the
                quantile and it is displayed
            samples (int): the number of samples of the stochastic process that are generated
            distribution (bool): whether it returns the log predictive function
            prior (bool): whether the prediction considers the prior
            noise (bool): wheter the prediction considers noise
            simulations (int): the number of simulation for the aproximation of the value of the
            stadistics

        Returns:
            Returns a dictionary which contains the information of the mean, std, var, cov, median,
            quantiles, quantiles_noise and distribution whether they are required.
        """
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

        n_simulations = 1
        if type(simulations) is int:
            n_simulations = simulations
            simulations = self.sampler(params, space, inputs, outputs, prior=prior, noise=noise, samples=simulations)
        values = DictObj()
        if mean:
            values['mean'] = self.mean(params, space, inputs, outputs, prior=prior, noise=noise, simulations=simulations)
        if var:
            values['variance'] = self.variance(params, space, inputs, outputs, prior=prior, noise=noise, simulations=simulations)
        if std:
            values['std'] = self.std(params, space, inputs, outputs, prior=prior, noise=noise, simulations=simulations)
        if cov:
            values['covariance'] = self.covariance(params, space, inputs, outputs, prior=prior, noise=noise)
        if median:
            values['median'] = self.median(params, space, inputs, outputs, prior=prior, noise=noise, simulations=simulations)
        if quantiles:
            values['quantile_up'] = self.quantiler(params, space, inputs, outputs, q=0.975, prior=prior, noise=noise, simulations=simulations)
            values['quantile_down'] = self.quantiler(params, space, inputs, outputs, q=0.025, prior=prior, noise=noise, simulations=simulations)
        if quantiles_noise:
            simulations_noise = self.sampler(params, space, inputs, outputs, prior=prior, noise=True, samples=n_simulations)
            values['noise_std'] = self.std(params, space, inputs, outputs, prior=prior, noise=True, simulations=simulations_noise)
            values['noise_up'] = self.quantiler(params, space, inputs, outputs, q=0.975, prior=prior, noise=True, simulations=n_simulations)
            values['noise_down'] = self.quantiler(params, space, inputs, outputs, q=0.025, prior=prior, noise=True, simulations=n_simulations)
        if samples > 0:
            values['samples'] = self.sampler(params, space, inputs, outputs, samples=samples, prior=prior, noise=noise)
        if distribution:
            #values['logp'] = lambda x: self.compiles['posterior_logp'](x, space, inputs, outputs, **params)
            #values['logpred'] = lambda x: self.compiles['posterior_logpred'](x, space, inputs, outputs, **params)
            values['logpredictive'] = lambda x: self.logpredictive(params, space, inputs, outputs, vector=x, prior=prior, noise=True)
        return values

    #TODO: Vectorized
    def logp_chain(self, chain, prior=False):
        out = np.empty(len(chain))
        for i in range(len(out)):
            out[i] = self.logp(chain[i], array=True, prior=prior)
        return out

    #@jit
    def fixed_logp(self, sampling_params, return_array=False):
        self.active.fixed_chain[:, self.active.sampling_dims] = sampling_params
        r = np.zeros(len(self.active.fixed_chain))
        for i, p in enumerate(self.active.fixed_chain):
            r[i] = self.compiles.array_posterior_logp(p, self.space, self.inputs, self.outputs)
        if return_array:
            return r
        else:
            return np.mean(r)

    #@jit
    def fixed_dlogp(self, sampling_params, return_array=False):
        self.active.fixed_chain[:, self.active.sampling_dims] = sampling_params
        r = list()
        for i, p in enumerate(self.active.fixed_chain):
            r.append(self.compiles.array_posterior_dlogp(p, self.space, self.inputs, self.outputs)[self.active.sampling_dims])
        if return_array:
            return np.array(r)
        else:
            return np.mean(np.array(r), axis=0)

    #@jit
    def fixed_loglike(self, sampling_params, return_array=False):
        self.active.fixed_chain[:, self.active.sampling_dims] = sampling_params
        r = np.zeros(len(self.active.fixed_chain))
        for i, p in enumerate(self.active.fixed_chain):
            r[i] = self.compiles.array_posterior_loglike(p, self.space, self.inputs, self.outputs)
        if return_array:
            return r
        else:
            return np.mean(r)

    #@jit
    def fixed_logprior(self, sampling_params, return_array=False):
        self.active.fixed_chain[:, self.active.sampling_dims] = sampling_params
        r = np.zeros(len(self.active.fixed_chain))
        for i, p in enumerate(self.active.fixed_chain):
            r[i] = self.compiles.array_prior_logp(p, self.space, self.inputs, self.outputs)
        if return_array:
            return r
        else:
            return np.mean(r)

    def find_MAP(self, start=None, points=1, return_points=False, plot=False, display=True,
                 powell=True, bfgs=True, init='bfgs', max_time=None):
        """
        This function calculates the Maximun A Posteriori alternating the bfgs and powell algorithms,

        Args:
            start (g3py.libs.DictObj): The initial parameters to start the optimization.
                The default value correspond to the default parameters of the gp. This could be a list
                of initial points.
            points (int): the number of (meta) iterations of the optimization problem
            return_points (bool): Determines whether the parameters points of the optimization
                are displayed.
            plot (bool): Determines whether the result it is plotted.
            display (bool): Determines whether the information of the optimization is displayed.
            powell (bool): Whether the powell algotithm it is used
            bfgs (bool): Whether the bfgs algotithm it is used
            init (str): The algorith with which it starts in the first iteration.
            max_time (int): the maximum number of seconds for every step in the optimization

        Returns:
            This function returns the optimal parameters of the loglikelihood function.
        """
        points_list = list()
        if start is None:
            start = self.params
        if self.active.fixed_datatrace is None:
            logp = lambda p: self.compiles.array_posterior_logp(p, self.space, self.inputs, self.outputs)
            dlogp = lambda p: self.compiles.array_posterior_dlogp(p, self.space, self.inputs, self.outputs)
        else:
            logp = self.fixed_logp
            dlogp = self.fixed_dlogp
        try:
            dlogp(self.active.sampling_params(start))
        except Exception as m:
            print(m)
            dlogp = None

        if type(start) is list:
            i = 0
            for s in start:
                i += 1
                points_list.append(('start' + str(i), logp(self.active.sampling_params(s)), s))
        else:
            points_list.append(('start', logp(self.active.sampling_params(start)), start))
        n_starts = len(points_list)
        if self.outputs is None:  # .get_value()
            print('For find_MAP it is necessary to have observations')
            return start
        if display:
            print('Starting function value (-logp): ' + str(-logp(self.active.sampling_params(points_list[0][2]))))
        if plot:
            plt.figure(0)
            self.plot(params=points_list[0][2], title='start')
            plt.show()
        if init is 'bfgs':
            check = 0
        else:
            check = 1
        with self.model:
            i = -1
            points -= 1
            while i < points:
                i += 1
                #try:
                if powell:
                    name, _, start = points_list[i // 2]
                else:
                    name, _, start = points_list[i]
                if (i % 2 == check or not powell) and bfgs:  #
                    if name.endswith('_bfgs'):
                        if i > n_starts:
                            points += 1
                        continue
                    name += '_bfgs'
                    if display:
                        print(name)
                    new = optimize(logp=logp, start=self.active.sampling_params(start), dlogp=dlogp, fmin='bfgs',
                                   max_time=max_time, disp=display)
                else:
                    if name.endswith('_powell'):
                        if i > n_starts:
                            points += 1
                        continue
                    name += '_powell'
                    if display:
                        print(name)
                    new = optimize(logp=logp, start=self.active.sampling_params(start), fmin='powell', max_time=max_time,
                                   disp=display)
                points_list.append((name, logp(new), self.active.dict_from_sampling_array(new)))
                if plot:
                    plt.figure(i + 1)
                    self.plot(params=self.active.dict_from_sampling_array(new), title=name)
                    plt.show()
                #except Exception as error:
                #    print(error)
                #    pass

        optimal = points_list[0]
        for test in points_list:
            if test[1] > optimal[1]:
                optimal = test
        _name, _ll, params = optimal
        params = DictObj(params)
        if display:
            print('find_MAP', params)
        if return_points is False:
            return params
        else:
            return params, points_list

    def sample_hypers(self, start=None, samples=1000, chains=None, ntemps=None, raw=False, noise_mult=0.1, noise_sum=0.01,
                      burnin_tol=0.001, burnin_method='multi-sum', outlayer_percentile=0.0005, clusters=None, prior=False, parallel=False, threads=1,
                      plot=False, file=None, load=True):
        """
        This function find the optimal hyperparameters of the logpredictive function using the
        'Ensemble MCMC' algorithm.
        Args:
            start (g3py.libs.DictObj): The initial parameters for the optimization. If start is None,
                it starts with the parameters obtained using find_MAP algorithm.
            samples (int): the number of iterations performed by the algorithm
            chains (int): the number of markov chains used in the sampling. The number of chains needs
                to be an even number and more than twice the dimension of the parameter space.
            ntemps (int): the number of temperatures used.
            raw (bool): this argument determines whether the result returned is raw or is pre-processed
            noise_mult (float): the variance of the multiplicative noise
            noise_sum (float): the variance of the aditive noise
            burnin_tol (float): It is the tolerance for the burnin.
            burnin_method (str): This set the algorith used to calculates the burnin
            outlayer_percentile (float): this takes a value between 0 and 1, and represent the value
                of the percentile to let out as outlayers.
            clusters (int): the number of clusters in which the sample is divided
            prior (bool): Whether the prior its considered
            parallel (bool): Whether the algorithm works in paralell or not.
            threads (int): the number of process to paralelize the algorithm
            plot (bool): whether the information of the datatrace are plotted or not.
            file (str): a path for save the datatrace
            load (bool): if load is True, a datatrace will be searched in the path given by file

        Returns:
            This function returns the information given by the Ensemble Markov Chain Monte Carlo Algorithm
            The information could be given tranformed or raw, depending of the boolean 'raw'.
            In the raw case, the information given contains evolution of each chain (which contains
             the parameters) across the iterations and the value of the loglikelihood in each iteration.
            Otherwhise, the function returns a datatrace, whose columns contains the values of
            every parameter, it transformation (logaritm transformation), the chain number to which it
            belong, the iteration number, and the 'burnin' and the 'outlayer' booleans.
        """
        ndim = len(self.active.sampling_dims)
        if chains is None:
            chains = 2*ndim
        if file is not None and load:
            try:
                datatrace = load_datatrace(file)
                if datatrace is not None:
                    if (datatrace._niter.max() == samples-1) and (datatrace._nchain.max() == chains-1):
                        if plot:
                            plot_datatrace(datatrace)
                        return datatrace
            except Exception as m:
                pass
        if start is None:
            start = self.find_MAP(display=False)
        if isinstance(start, dict):
            start = self.active.dict_to_array(start)

        if len(start.shape) == 1:
            start = start[self.active.sampling_dims]
        elif len(start.shape) == 2:
            start = start[:, self.active.sampling_dims]
        elif len(start.shape) == 3:
            start = start[:, :, self.active.sampling_dims]
        if self.active.fixed_datatrace is None:
            if ntemps is None:
                if prior is False:
                    logp = lambda p: self.compiles.array_posterior_logp(p, self.space, self.inputs, self.outputs)
                else:
                    logp = lambda p: self.compiles.array_prior_logp(p, self.space, self.inputs, self.outputs)
                loglike = None
                logprior = None
            else:
                logp = None
                logprior = lambda p: self.compiles.array_prior_logp(p, self.space, self.inputs, self.outputs)
                if prior is False:
                    loglike = lambda p: self.compiles.array_posterior_loglike(p, self.space, self.inputs, self.outputs)
                else:
                    loglike = lambda p: zero32
        else:
            if ntemps is None:
                if prior is False:
                    logp = self.fixed_logp
                else:
                    logp = self.fixed_logprior
                loglike = None
                logprior = None
            else:
                logp = None
                if prior is False:
                    loglike = self.fixed_loglike
                else:
                    loglike = lambda p: zero32
                logprior = self.fixed_logprior

        def parallel_mcmc(nchains):
            return mcmc_ensemble(ndim, samples=samples, chains=nchains, ntemps=ntemps, start=start,
                                           logp=logp, loglike=loglike, logprior=logprior,
                                           noise_mult=noise_mult, noise_sum=noise_sum, threads=threads)

        if parallel in [None, 0, 1]:
            lnprob, echain = parallel_mcmc(nchains=chains)
        else:
            import multiprocessing as mp
            p = mp.Pool(parallel)
            r = p.map(parallel_mcmc, list([chains/parallel]*parallel))
            lnprob, echain = [], []
            for k in r:
                lk, le = k
                lnprob = np.concatenate([lnprob, lk])
                echain = np.concatenate([echain, le])

        complete_chain = np.empty((echain.shape[0], echain.shape[1], self.ndim))
        complete_chain[:, :, self.active.sampling_dims] = echain
        if self.active.fixed_datatrace is not None:
            print("TODO: Check THIS complete_chain with MEAN")
            complete_chain[:, :, self.active.fixed_dims] = self.active.fixed_chain[:, self.active.fixed_dims].mean(axis=0)
        if raw:
            return complete_chain, lnprob
        else:
            datatrace = chains_to_datatrace(self, complete_chain, ll=lnprob, burnin_tol=burnin_tol,
                                       burnin_method=burnin_method, burnin_dims=self.active.sampling_dims,
                                       outlayer_percentile=outlayer_percentile, clusters=clusters)
            if file is not None:
                save_datatrace(datatrace, file)
            if plot:
                plot_datatrace(datatrace)
            return datatrace

    @property
    def ndim(self):
        return self.active.ndim


