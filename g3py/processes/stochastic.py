import numpy as np
import pymc3 as pm
import scipy as sp
import theano as th
from g3py.functions import Mean, Kernel, Mapping, KernelSum, WN, tt_to_num, zeros, def_space, debug, trans_hypers, Identity
from g3py.libs import tt_to_cov, cholesky_robust, makefn, text_plot
from g3py.models import Model, TGPDist, ConstantStep, RobustSlice
from ipywidgets import interact
from matplotlib import pyplot as plt
from theano import tensor as tt
from theano.tensor import slinalg as sL
from theano.tensor import nlinalg as nL


class StochasticProcess:
    """Abstract class used to define a StochasticProcess.

    Attributes:
        model (pm.Model): Reference to the context pm.Model
    """
    def __init__(self, space=1, location: Mean=None, kernel: Kernel=None, mapping: Mapping=None, noise=True, freedom=None,
                 name=None, inputs=None, outputs=None, hidden=None, description=None):
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
        self.model = Model.get_context()

        # Space, Hidden, Observed
        __, self.space_values, self.space_index = def_space(space)
        __, self.inputs_values, self.observed_index = def_space(inputs)
        __, self.outputs_values, __ = def_space(outputs, squeeze=True)
        self.space = tt.matrix(self.name + '_space', dtype=th.config.floatX)
        self.inputs = tt.matrix(self.name + '_inputs', dtype=th.config.floatX)
        self.outputs = tt.vector(self.name + '_outputs', dtype=th.config.floatX)
        self.space.tag.test_value = self.space_values
        self.inputs.tag.test_value = self.inputs_values
        self.outputs.tag.test_value = self.outputs_values
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
            self.location.check_hypers(self.name + '_')
            self.kernel.check_hypers(self.name + '_')
            self.mapping.check_hypers(self.name + '_')

        print('Space Dimensions: ', self.space_values.shape)
        print('Inputs Dimensions: ', self.inputs_values.shape)
        print('Output Dimensions: ', self.outputs_values.shape)

        # Hyper-parameters values
        self.params_current = None
        self.params_widget = None
        self.params_fixed = {}

        # Basic Tensors
        self.location_space = self.location(self.space)
        self.location_inputs = self.location(self.inputs)

        self.kernel_space = tt_to_cov(self.kernel.cov(self.space))
        self.kernel_inputs = tt_to_cov(self.kernel.cov(self.inputs))
        self.kernel_space_inputs = tt_to_num(self.kernel.cov(self.space, self.inputs))

        self.kernel_f_space = tt_to_cov(self.kernel_f.cov(self.space))
        self.kernel_f_inputs = tt_to_cov(self.kernel_f.cov(self.inputs))
        self.kernel_f_space_inputs = tt_to_num(self.kernel_f.cov(self.space, self.inputs))

        self.mapping_outputs = tt_to_num(self.mapping.inv(self.outputs))

        # Prior
        self.prior_mean = None
        self.prior_covariance = None
        self.prior_variance = None
        self.prior_noise = None
        self.prior_quantile_up = None
        self.prior_quantile_down = None
        self.prior_noise_up = None
        self.prior_noise_down = None

        # Posterior
        self.posterior_mean = None
        self.posterior_covariance = None
        self.posterior_variance = None
        self.posterior_noise = None
        self.posterior_quantile_up = None
        self.posterior_quantile_down = None
        self.posterior_noise_up = None
        self.posterior_noise_down = None

        self.distribution = None

        self.compiles = {}

    def define_process(self):
        pass

    def compile(self):
        self.define_process()
        print('Compiling')

        params = [self.space] + self.model.vars
        self.compiles['prior_mean'] = makefn(params, self.prior_mean)
        self.compiles['prior_covariance'] = makefn(params, self.prior_covariance)
        self.compiles['prior_variance'] = makefn(params, self.prior_variance)
        self.compiles['prior_noise'] = makefn(params, self.prior_noise)
        self.compiles['prior_quantile_up'] = makefn(params, self.prior_quantile_up)
        self.compiles['prior_quantile_down'] = makefn(params, self.prior_quantile_down)
        self.compiles['prior_noise_up'] = makefn(params, self.prior_noise_up)
        self.compiles['prior_noise_down'] = makefn(params, self.prior_noise_down)

        params = [self.space, self.inputs, self.outputs] + self.model.vars
        self.compiles['posterior_mean'] = makefn(params, self.posterior_mean)
        self.compiles['posterior_covariance'] = makefn(params, self.posterior_covariance)
        self.compiles['posterior_variance'] = makefn(params, self.posterior_variance)
        self.compiles['posterior_noise'] = makefn(params, self.posterior_noise)
        self.compiles['posterior_quantile_up'] = makefn(params, self.posterior_quantile_up)
        self.compiles['posterior_quantile_down'] = makefn(params, self.posterior_quantile_down)
        self.compiles['posterior_noise_up'] = makefn(params, self.posterior_noise_up)
        self.compiles['posterior_noise_down'] = makefn(params, self.posterior_noise_down)

        return

        params = self.model.vars
        random, sampler = self.sampler_gp()
        sampler_gp = makefn([random] + params, sampler)
        random, sampler = self.sampler_tgp()
        sampler_tgp = makefn([random] + params, sampler)
        self.compiles['sampler_gp'] = sampler_gp
        self.compiles['sampler_tgp'] = sampler_tgp


        params = self.model.vars
        value, dist = self.marginal_gp()

        dist_gp = makefn([value] + params, dist[index])

        trans = makefn([value] + params, self.mapping(value))
        trans_inv = makefn([value] + params, self.mapping.inv(value))
        det_jac_trans_inv = makefn([value] + params, tt.exp(self.mapping.logdet_dinv(value)))

        value, dist = self.marginal_tgp()
        dist_tgp = makefn([value] + params, dist[index])

        self.compiles['trans'] = trans
        self.compiles['trans_inv'] = trans_inv
        self.compiles['det_jac_trans_inv'] = det_jac_trans_inv
        self.compiles['dist_gp'] = dist_gp
        self.compiles['dist_tgp'] = dist_tgp

    def describe(self, title=None, x=None, y=None, text=None):
        if title is not None:
            self.description['title'] = title
        if title is not None:
            self.description['x'] = x
        if title is not None:
            self.description['y'] = y
        if title is not None:
            self.description['text'] = text

    def set_space(self, space):
        new_space, self.space_values, self.space_index = def_space(space, self.name + '_space')
        self.space.set_value(new_space.get_value())

    def observed(self, inputs, outputs):
        __, self.inputs_values, self.observed_index = def_space(inputs)
        __, self.outputs_values, __ = def_space(outputs, squeeze=True)


    def fix_params(self, fixed_params):
        self.params_fixed = fixed_params

    def check_params_dims(self, **params):
        r = dict()
        for k, v in params.items():
            r[k] = np.array(v, dtype=th.config.floatX).reshape(self.model[k].tag.test_value.shape)
        return r

    def def_process(self):
        pass

    def marginal(self):
        pass

    def prior(self, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False):
        pass

    def posterior(self, mean=True, var=True, cov=False, median=False, quantiles=False, noise=False):
        pass

    def subprocess(self, subkernel, mean=True, cov=False, var=True, median=False, quantiles=False, noise=False):
        pass

    def sampler(self, space=None, params=None, samples=1):
        pass

    def predict(self, space=None, params=None, mean=True, var=True, median=False, quantiles=False, noise=False):
        pass

    def plot(self, space=None, params=None, mean=True, var=True, median=False, quantiles=False, noise=False, samples=0,
             big=True, scores=False, title=None, loc=1):
        pass

    def plot_widget(self, **params):
        params = self.check_params_dims(**params)
        self._widget_params = params
        self.plot_data(big=True)
        self.plot_tgp_quantiles(params)
        self.plot_tgp_moments(params)
        #self.description['scores'] = self.scores(params)
        text_plot(self.description['title'],  self.description['x'],  self.description['y'], loc=1)
        #self.plot_model(**params)

    def widget_params(self, params=None, space=None):
        if params is None:
            params = self.get_params()
        intervals = dict()
        for k, v in params.items():
            v = np.squeeze(v)
            if v > 0.1:
                intervals[k] = [0, 2*v]
            elif v < -0.1:
                intervals[k] = [2*v, 0]
            else:
                intervals[k] = [-5.00, 5.00]
        interact(self.plot_widget, __manual=True, **intervals)

    def widget_trace(self, trace, space=None):
        self.params_widget = trace
        self.get_point(self.params_widget, id_trace)
        interact(self.plot_widget, __manual=True, id_trace=[0, len(self.params_widget) - 1])


    @property
    def fixed_vars(self):
        return [t for t in self.model.vars if t.name in self.params_fixed.keys()]

    @property
    def sampling_vars(self):
        return [t for t in self.model.vars if t not in self.fixed_vars]

    def default_hypers(self):
        x = self.inputs_values
        y = self.outputs_values
        return {**self.location.default_hypers_dims(x, y), **self.kernel.default_hypers_dims(x, y),
                **self.mapping.default_hypers_dims(x, y)}

    def get_params_dims(self, params):
        r = dict()
        for k, v in params.items():
            r[k] = np.array(v, dtype=th.config.floatX).reshape(self.model[k].tag.test_value.shape)
        return r

    def get_params_default(self, fixed=True):
        if self.inputs is None:
            return self.model.test_point
        default = {}
        for k, v in trans_hypers(self.default_hypers()).items():
            default[k.name] = v
        if fixed:
            default.update(self.params_fixed)
        return default

    def get_params_current(self, fixed=True):
        if len(self.params_current) == 0:
            return self.get_params_default()
        if fixed:
            self.params_current.update(self.params_fixed)
        return self.params_current

    def get_params_widget(self, fixed=True):
        if len(self.params_widget) == 0:
            return self.get_params_default()
        if fixed:
            self.params_widget.update(self.params_fixed)
        return self.params_widget

    def scores_params(self, params=None):
        try:
            logp_train = np.nanmean(self.model.logp(params))
        except:
            logp_train = np.float32(0)

        mean = self.compiles['m1'](**params)
        var = mean**2 - self.compiles['m2'](**params)
        #print(mean.shape)
        #print(self.outputs.get_value().shape)

        mse_train = 0#np.nanmean((mean - self.outputs.get_value()) ** 2 + var)
        bias_train = 0#np.nanmean(np.abs(mean - self.outputs.get_value()))

        self.swap_test_obs()
        try:
            logp_test = np.nanmean(self.model.logp(params))
        except:
            logp_test = np.float32(0)
        mean = self.compiles['m1'](**params)
        var = mean**2 - self.compiles['m2'](**params)
        mse_test = 0#np.nanmean((mean - self.outputs.get_value()) ** 2 + var)
        bias_test = 0#np.nanmean(np.abs(mean - self.outputs.get_value()))
        self.swap_test_obs()

        d = {'logp_train': logp_train,
             'mse_train': mse_train,
             'mab_train': bias_train,
             'logp_test': logp_test,
             'mse_test': mse_test,
             'mab_test': bias_test}
        return d

    def find_MAP(self, start=None, points=1, plot=False, return_points=False, display=True):
        points_list = list()
        if start is None:
            start = self.get_params_current()
        points_list.append(('start', self.model.logp(start), start))
        plt.figure(0)
        print('Starting function value (-logp): '+str(-self.model.logp(start)))
        self.plot_tgp(start, 'start')
        plt.show()
        with self.model:
            for i in range(points):
                try:
                    name, logp, start = points_list[i // 2]
                    if i % 2 == 0:
                        name += '_bfgs'
                        print('\n' + name)
                        new = pm.find_MAP(fmin=sp.optimize.fmin_bfgs, vars=self.sampling_vars, start=start)
                    else:
                        name += '_powell'
                        print('\n' + name)
                        new = pm.find_MAP(fmin=sp.optimize.fmin_powell, vars=self.sampling_vars, start=start)
                    points_list.append((name, self.model.logp(new), new))
                    if plot:
                        plt.figure(i+1)
                        self.plot_tgp(new, name)
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

    def sample_hypers(self, start=None, samples=1000, chains=1, advi=True):
        if start is None:
            start = self.find_default()
        with self.model:
            if advi:
                advi_mu, advi_sm, advi_elbo = pm.advi(vars=self.sampling_vars, start=start, n=20000) #OK
                for k, v in advi_sm.items():
                    advi_sm[k] = v**2
            else:
                advi_mu = start
                advi_sm = None
            if len(self.fixed_params) > 0:
                step = [ConstantStep(vars=self.fixed_vars)]  # Fue eliminado por error en pymc3
                step += [RobustSlice(vars=self.sampling_vars)]  # Slice original se cuelga si parte del óptimo
                #step += [pm.Metropolis(vars=self.sampling_vars, tune=False)] # OK
                #step += [pm.HamiltonianMC(vars=self.sampling_vars, scaling=advi_sm, is_cov=True)] #Scaling is not positive definite. Simple check failed. Diagonal contains negatives. Check indexes
                #step += [pm.NUTS(vars=self.sampling_vars, scaling=advi_sm, is_cov=True)] #BUG float32
            else:
                step = RobustSlice()
            trace = pm.sample(samples, step=step, start=advi_mu, njobs=chains)
        return trace

    def save_model(self, path, params=None):
        try:
            with self.model:
                with open(path, 'wb') as f:
                    pickle.dump((params), f, protocol=-1)
            print('Saved model '+path)
        except:
            print('Error saving model '+path)

    def plot_space(self):
        plt.plot(self.space_index, self.space_values)
        if self.observed_index is not None:
            plt.plot(self.observed_index, self.inputs_values)

    def plot_data(self, big=False):
        if big:
            self.plot_data_big()
        else:
            self.plot_data_normal()

    def plot_data_normal(self):
        if self.hidden is not None:
            plt.plot(self.space_index, self.hidden, label='Hidden Processes')
        if self.outputs_values is not None:
            plt.plot(self.observed_index, self.outputs_values, '.k', label='Observations')

    def plot_data_big(self):
        if self.hidden is not None:
            plt.plot(self.space_index, self.hidden, linewidth=4, label='Hidden Processes')
        if self.outputs_values is not None:
            plt.plot(self.observed_index, self.outputs_values, '.k', ms=20, label='Observations')

    def plot_concentration(self):
        return plt.matshow(self.cov_inputs)


def gauss_hermite(f, mu, sigma, a, w):
    return tt.dot(w, f(mu + sigma * np.sqrt(2) * a)) / np.sqrt(np.pi)